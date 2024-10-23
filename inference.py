from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from model2vec import StaticModel
from tqdm.autonotebook import tqdm

PROJ_DIM = 96

def pad_and_stack_vectors(
    vector_list: List[np.ndarray], 
    device: str = 'cpu',
    normalize: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pad and stack vectors into a single tensor, handling ColBERT's token-level embeddings.
    
    Args:
        vector_list: List of token embeddings arrays
        device: Device to place tensors on
        normalize: Whether to L2 normalize embeddings
        
    Returns:
        Tuple of (padded_embeddings, attention_mask)
    """
    max_seq_length = max(len(vector) for vector in vector_list)
    emb_dim = vector_list[0].shape[-1]
    
    # Verify embedding dimension is 128
    assert emb_dim == PROJ_DIM, f"Expected embedding dimension {PROJ_DIM}, got {emb_dim}"
    
    # Initialize tensors
    padded_tensor = torch.zeros(
        (len(vector_list), max_seq_length, emb_dim), 
        dtype=torch.float32, 
        device=device
    )
    attention_mask = torch.zeros(
        (len(vector_list), max_seq_length), 
        dtype=torch.bool, 
        device=device
    )
    
    # Fill tensors
    for i, vector in enumerate(vector_list):
        if isinstance(vector, np.ndarray):
            vector = torch.from_numpy(vector)
        vector = vector.to(device=device, dtype=torch.float32)
        
        # Normalize if requested
        if normalize:
            vector = F.normalize(vector, p=2, dim=-1)
            
        padded_tensor[i, :len(vector)] = vector
        attention_mask[i, :len(vector)] = True
    
    return padded_tensor, attention_mask

def compute_relevance_scores(
    query_embeddings: Union[torch.Tensor, np.ndarray],
    document_embeddings: torch.Tensor,
    document_mask: torch.Tensor,
    batch_size: int = 32,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute ColBERT-style relevance scores using MaxSim operation.
    
    Args:
        query_embeddings: Query token embeddings [num_query_terms, embedding_dim]
        document_embeddings: Document token embeddings [num_docs, max_doc_length, embedding_dim]
        document_mask: Boolean mask for valid document tokens [num_docs, max_doc_length]
        batch_size: Batch size for processing documents
        device: Device to compute scores on
        
    Returns:
        Tuple of (sorted document indices, relevance scores)
    """
    # Convert query to tensor if needed
    if isinstance(query_embeddings, np.ndarray):
        query_embeddings = torch.from_numpy(query_embeddings)
    
    # Move to device and ensure shapes
    query_embeddings = query_embeddings.to(device)
    if query_embeddings.ndim == 2:
        query_embeddings = query_embeddings.unsqueeze(0)
    
    # Normalize query embeddings
    query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
    
    # Process in batches
    total_docs = document_embeddings.size(0)
    all_scores = []
    
    for start_idx in range(0, total_docs, batch_size):
        end_idx = min(start_idx + batch_size, total_docs)
        batch_docs = document_embeddings[start_idx:end_idx]
        batch_mask = document_mask[start_idx:end_idx]
        
        # Compute similarity scores [batch_size, num_query_terms, max_doc_length]
        scores = torch.matmul(query_embeddings, batch_docs.transpose(1, 2))
        
        # Mask out padding tokens
        scores = scores.masked_fill(~batch_mask.unsqueeze(1), float('-inf'))
        
        # MaxSim operation: max similarity per query term [batch_size, num_query_terms]
        max_scores = scores.max(dim=-1).values
        
        # Sum across query terms [batch_size]
        doc_scores = max_scores.sum(dim=1)
        all_scores.append(doc_scores)
    
    # Combine all batches
    total_scores = torch.cat(all_scores)
    
    # Sort documents by score
    sorted_scores, sorted_indices = torch.sort(total_scores, descending=True)
    
    return sorted_indices, sorted_scores

def encode_batch(
    model: StaticModel,
    texts: List[str],
    batch_size: int = 32,
    show_progress: bool = True,
    desc: str = "Encoding"
) -> List[np.ndarray]:
    """Helper function to encode texts in batches."""
    embeddings = []
    iterator = range(0, len(texts), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc=desc)
        
    for start_idx in iterator:
        end_idx = min(start_idx + batch_size, len(texts))
        batch = texts[start_idx:end_idx]
        batch_emb = model.encode_as_sequence(batch)
        embeddings.extend(batch_emb)
    return embeddings

def encode_queries_and_documents(
    model: StaticModel,
    queries: List[str],
    documents: List[str],
    batch_size: int = 32,
    device: str = 'cpu',
    show_progress: bool = True
) -> Tuple[List[np.ndarray], torch.Tensor, torch.Tensor]:
    """
    Encode both queries and documents efficiently.
    
    Args:
        model: The StaticModel to use for encoding
        queries: List of query strings
        documents: List of document strings
        batch_size: Batch size for processing
        device: Device to use
        show_progress: Whether to show progress bars
    
    Returns:
        Tuple of (query_embeddings, document_embeddings, document_mask)
    """
    # Encode in batches
    query_embeddings = encode_batch(
        model, queries, batch_size, show_progress, desc="Encoding queries"
    )
    doc_embeddings = encode_batch(
        model, documents, batch_size, show_progress, desc="Encoding documents"
    )
    
    # Verify dimensions
    for emb in query_embeddings + doc_embeddings:
        assert emb.shape[-1] == PROJ_DIM, f"Expected {PROJ_DIM}D embeddings, got {emb.shape[-1]}"
    
    # Pad and stack documents
    doc_tensor, doc_mask = pad_and_stack_vectors(
        doc_embeddings, 
        device=device,
        normalize=True
    )
    
    return query_embeddings, doc_tensor, doc_mask

def search(
    model: StaticModel,
    query: str,
    documents: List[str],
    top_k: Optional[int] = None,
    batch_size: int = 32,
    device: str = None,
    show_progress: bool = True
) -> List[Tuple[int, float]]:
    """
    Search documents for the given query.
    
    Args:
        model: The StaticModel to use
        query: Query string
        documents: List of documents to search
        top_k: Number of results to return (None for all)
        batch_size: Batch size for processing
        device: Device to use
        show_progress: Whether to show progress
        
    Returns:
        List of (document_idx, score) tuples sorted by score
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    # Encode query and documents
    query_emb = model.encode_as_sequence(query)
    doc_embeddings, doc_mask = pad_and_stack_vectors(
        [model.encode_as_sequence(d) for d in tqdm(documents, disable=not show_progress)],
        device=device,
        normalize=True
    )
    
    # Get scores
    sorted_indices, scores = compute_relevance_scores(
        query_emb,
        doc_embeddings,
        doc_mask,
        batch_size=batch_size,
        device=device
    )
    
    # Convert to CPU numpy
    sorted_indices = sorted_indices.cpu().numpy()
    scores = scores.cpu().numpy()
    
    # Return top-k results
    if top_k is not None:
        sorted_indices = sorted_indices[:top_k]
        scores = scores[:top_k]
        
    return list(zip(sorted_indices, scores))

if __name__ == "__main__":
    # Example usage
    model_name = "./m2v_model"
    model = StaticModel.from_pretrained(model_name)
    
    documents = [
        "It's dangerous to go alone!",
        "Take this sword with you.",
        "The princess is in another castle.",
        "All your base are belong to us."
    ]
    
    query = "What is dangerous?"
    
    # Simple search
    results = search(model, query, documents, top_k=2)
    
    print("\nSearch Results:")
    for idx, score in results:
        print(f"Score: {score:.4f} | Document: {documents[idx]}")
        
    # Or use the lower-level API for more control
    query_emb = model.encode_as_sequence(query)
    doc_embs, doc_mask = pad_and_stack_vectors(
        [model.encode_as_sequence(d) for d in documents],
        normalize=True
    )
    
    sorted_idx, scores = compute_relevance_scores(
        query_emb, 
        doc_embs, 
        doc_mask
    )
    
    print("\nDirect Scoring Results:")
    for idx, score in zip(sorted_idx, scores):
        print(f"Score: {score:.4f} | Document: {documents[idx]}")