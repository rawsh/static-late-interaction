import json
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from model2vec.distill.distillation import _post_process_embeddings
from model2vec.distill.tokenizer import remove_tokens
from model2vec.distill.utils import select_optimal_device
from model2vec.model import StaticModel
from safetensors import safe_open
from safetensors.torch import load_model as load_safetensors_model
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Dense as DenseSentenceTransformer
from sentence_transformers.util import import_from_string
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from transformers.utils import cached_file


class Dense(DenseSentenceTransformer):
    """Performs linear projection on the token embeddings to a lower dimension.

    Parameters
    ----------
    in_features
        Size of the embeddings in output of the tansformer.
    out_features
        Size of the output embeddings after linear projection
    bias
        Add a bias vector
    init_weight
        Initial value for the matrix of the linear layer
    init_bias
        Initial value for the bias of the linear layer.

    Examples
    --------
    >>> from pylate import models

    >>> model = models.Dense(
    ...     in_features=768,
    ...     out_features=128,
    ... )

    >>> features = {
    ...     "token_embeddings": torch.randn(2, 768),
    ... }

    >>> projected_features = model(features)

    >>> assert projected_features["token_embeddings"].shape == (2, 128)
    >>> assert isinstance(model, DenseSentenceTransformer)

    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation_function=torch.nn.Identity(),
        init_weight: torch.Tensor = None,
        init_bias: torch.Tensor = None,
    ) -> None:
        super(Dense, self).__init__(
            in_features, out_features, bias, activation_function, init_weight, init_bias
        )

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Performs linear projection on the token embeddings."""
        token_embeddings = features["token_embeddings"]
        projected_embeddings = self.linear(token_embeddings)
        features["token_embeddings"] = projected_embeddings
        return features

    @staticmethod
    def from_sentence_transformers(dense: DenseSentenceTransformer) -> "Dense":
        """Converts a SentenceTransformer Dense model to a Dense model.
        Our Dense model does not have the activation function.
        """
        config = dense.get_config_dict()
        config["activation_function"] = torch.nn.Identity()
        model = Dense(**config)
        model.load_state_dict(dense.state_dict())
        return model

    @staticmethod
    def from_stanford_weights(
        model_name_or_path: str | os.PathLike,
        cache_folder: str | os.PathLike | None = None,
        revision: str | None = None,
        local_files_only: bool | None = None,
        token: str | bool | None = None,
        use_auth_token: str | bool | None = None,
    ) -> "Dense":
        """Load the weight of the Dense layer using weights from a stanford-nlp checkpoint.

        Parameters
        ----------
        model_name_or_path (`str` or `os.PathLike`):
            This can be either:
            - a string, the *model id* of a model repo on huggingface.co.
            - a path to a *directory* potentially containing the file.
        cache_folder (`str` or `os.PathLike`, *optional*):
            Path to a directory in which a downloaded pretrained model configuration should be cached if the standard
            cache should not be used.
        token (`str` or *bool*, *optional*):
            The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
            when running `huggingface-cli login` (stored in `~/.huggingface`).
        revision (`str`, *optional*, defaults to `"main"`):
            The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
            git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
            identifier allowed by git.
        local_files_only (`bool`, *optional*, defaults to `False`):
            If `True`, will only try to load the tokenizer configuration from local files.
        """
        # Check if the model is locally available
        if not (os.path.exists(os.path.join(model_name_or_path))):
            # Else download the model/use the cached version
            model_name_or_path = cached_file(
                model_name_or_path,
                filename="model.safetensors",
                cache_dir=cache_folder,
                revision=revision,
                local_files_only=local_files_only,
                token=token,
                use_auth_token=use_auth_token,
            )
        # If the model a local folder, load the safetensor
        else:
            model_name_or_path = os.path.join(model_name_or_path, "model.safetensors")
        with safe_open(model_name_or_path, framework="pt", device="cpu") as f:
            state_dict = {"linear.weight": f.get_tensor("linear.weight")}

        # Determine input and output dimensions
        in_features = state_dict["linear.weight"].shape[1]
        out_features = state_dict["linear.weight"].shape[0]

        # Create Dense layer instance
        model = Dense(in_features=in_features, out_features=out_features, bias=False)

        model.load_state_dict(state_dict)
        return model

    @staticmethod
    def load(input_path) -> "Dense":
        """Load a Dense layer."""
        with open(os.path.join(input_path, "config.json")) as fIn:
            config = json.load(fIn)

        config["activation_function"] = import_from_string(
            config["activation_function"]
        )()

        model = Dense(**config)

        if os.path.exists(os.path.join(input_path, "model.safetensors")):
            load_safetensors_model(model, os.path.join(input_path, "model.safetensors"))
            return model

        model.load_state_dict(
            torch.load(
                os.path.join(input_path, "pytorch_model.bin"),
                map_location=torch.device("cpu"),
            )
        )
        return model


class ColbertDistiller:
    def __init__(
        self,
        model_name: str,
        embedding_size: int = 128,
        q_token_id: int = 1,
        d_token_id: int = 2,
        device: Optional[str] = None,
    ):
        """Initialize ColBERT distiller."""
        self.model_name = model_name
        self.q_token_id = q_token_id
        self.d_token_id = d_token_id
        self.device = select_optimal_device(device)
        self.embedding_size = embedding_size
        
        # First try to load as a stanford-nlp ColBERT model
        try:
            # Load base model
            self.base_model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Check if it's a stanford-nlp model
            if (hasattr(self.base_model, 'config') and 
                hasattr(self.base_model.config, 'architectures') and 
                self.base_model.config.architectures is not None and 
                self.base_model.config.architectures[0] == "HF_ColBERT"):
                
                # Load Dense layer from stanford weights
                self.projection = Dense.from_stanford_weights(
                    self.model_name,
                    local_files_only=True
                ).to(self.device)
                print("Loaded Stanford-NLP ColBERT model")
                
            else:
                # Try loading as SentenceTransformer
                sent_transformer = SentenceTransformer(model_name)
                if len(sent_transformer._modules) >= 2:
                    # Get the Dense layer
                    dense_module = sent_transformer._modules['1']
                    if isinstance(dense_module, Dense):
                        self.projection = dense_module
                    else:
                        self.projection = Dense.from_sentence_transformers(dense_module)
                    print("Loaded SentenceTransformer ColBERT model")
                else:
                    raise ValueError("Model does not contain projection layer")
                    
        except Exception as e:
            print(f"Failed to load model: {str(e)}")
            raise ValueError("Could not load ColBERT model properly")
            
        # Verify projection dimension
        if self.projection.out_features != self.embedding_size:
            raise ValueError(f"Expected projection dimension {self.embedding_size}, got {self.projection.out_features}")

    def forward_pass(self, batch: torch.Tensor) -> torch.Tensor:
        """Process a batch of tokens through BERT and projection layer."""
        with torch.no_grad():
            # Get BERT outputs
            encoded = self.base_model(
                input_ids=batch,
                attention_mask=torch.ones_like(batch),
                token_type_ids=torch.zeros_like(batch),
                return_dict=True
            )
            
            # Get the embeddings for the target tokens
            token_embeddings = encoded.last_hidden_state[:, 2]
            
            # Project using Dense layer
            features = {"token_embeddings": token_embeddings}
            projected = self.projection(features)["token_embeddings"]
            
            # Normalize
            if projected.dtype == torch.bfloat16:
                projected = projected.float()
            normalized = torch.nn.functional.normalize(projected, p=2, dim=-1)
            
            return normalized

    def create_colbert_embeddings(self) -> Tuple[List[str], np.ndarray]:
        """Create embeddings for all tokens in the vocabulary."""
        # Get token IDs
        ids = torch.arange(self.tokenizer.vocab_size)

        # Get BOS/EOS tokens from a dummy encoding
        dummy_encoding = self.tokenizer.encode("A")
        eos_token_id, bos_token_id = dummy_encoding[0], dummy_encoding[-1]

        # Create tensors for special tokens
        eos = torch.full([len(ids)], fill_value=eos_token_id)
        bos = torch.full([len(ids)], fill_value=bos_token_id)
        document_markers = torch.full([len(ids)], fill_value=self.d_token_id)

        # Stack into sequences: [BOS, DOC_MARKER, TOKEN, EOS]
        stacked = torch.stack([bos, document_markers, ids, eos], dim=1)
        print(f"Input sequence shape: {stacked.shape}")

        # Process in batches
        intermediate_weights: List[np.ndarray] = []
        batch_size = 1024
        
        for batch_idx in tqdm(range(0, len(stacked), batch_size)):
            batch = stacked[batch_idx:batch_idx + batch_size].to(self.device)
            normalized = self.forward_pass(batch)
            intermediate_weights.append(normalized.cpu().numpy())
            
        out_weights = np.concatenate(intermediate_weights)
        print(f"Final embedding shape: {out_weights.shape}")
        
        assert out_weights.shape[1] == self.embedding_size, \
            f"Expected embedding dimension {self.embedding_size}, got {out_weights.shape[1]}"

        # Get tokens
        tokens = self.tokenizer.convert_ids_to_tokens(ids)
        return tokens, out_weights

    def distill(self) -> StaticModel:
        """Distill the ColBERT model into a static model."""
        print(f"Distilling ColBERT model: {self.model_name}")
        
        # Create embeddings
        tokens, embeddings = self.create_colbert_embeddings()

        # Remove unused tokens
        wrong_tokens = [x for x in tokens if x.startswith("[unused")]
        vocab = self.tokenizer.get_vocab()
        wrong_token_ids = [vocab[x] for x in wrong_tokens]
        
        # Remove from tokenizer
        new_tokenizer = remove_tokens(self.tokenizer.backend_tokenizer, wrong_tokens)
        
        # Remove the embeddings for the unused tokens
        embeddings = np.delete(embeddings, wrong_token_ids, axis=0)
        print(f"Removed {len(wrong_tokens)} unused tokens from the vocabulary")
        
        # Apply only Zipf weighting, skip PCA
        processed_embeddings = _post_process_embeddings(
            embeddings,
            pca_dims=None,  # Skip PCA to preserve ColBERT space
            apply_zipf=True
        )
        
        # Create config
        config = {
            "tokenizer_name": self.model_name,
            "apply_pca": None,
            "apply_zipf": True,
            "hidden_dim": processed_embeddings.shape[1],
            "seq_length": 1000000,  # No sequence length limit
        }
        
        return StaticModel(
            vectors=processed_embeddings,
            tokenizer=new_tokenizer,
            config=config,
            base_model_name=self.model_name
        )


def distill_colbert(
    model_name: str,
    embedding_size: int = 128,
    q_token_id: int = 1,
    d_token_id: int = 2,
    device: Optional[str] = None
) -> StaticModel:
    """
    Distill a ColBERT model into a static model2vec representation.
    
    Args:
        model_name: Name of the ColBERT model to distill
        embedding_size: Size of embeddings (default: 128)
        q_token_id: Query marker token id (default: 1)
        d_token_id: Document marker token id (default: 2)
        device: Device to run on (default: None, will use CUDA if available)
    
    Returns:
        StaticModel: The distilled model
    """
    distiller = ColbertDistiller(
        model_name=model_name,
        embedding_size=embedding_size,
        q_token_id=q_token_id,
        d_token_id=d_token_id,
        device=device
    )
    return distiller.distill()


if __name__ == "__main__":
    model_name = "answerdotai/answerai-colbert-small-v1"
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.mps.is_available() else 'cpu')
    m2v_model = distill_colbert(model_name=model_name, embedding_size=96, device=device)
    m2v_model.save_pretrained("m2v_model")