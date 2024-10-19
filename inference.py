from model2vec import StaticModel
import torch

def pad_and_stack_vectors(vector_list):
    max_seq_length = max(len(vector) for vector in vector_list)
    emb_dim = vector_list[0].shape[-1]
    pad_value = 0  # or any other appropriate padding value

    # Convert numpy dtype to PyTorch dtype
    torch_dtype = torch.float32  # Assuming the original dtype was float32

    padded_tensor = torch.full((len(vector_list), max_seq_length, emb_dim), pad_value, dtype=torch_dtype)
    
    for i, vector in enumerate(vector_list):
        padded_tensor[i, :len(vector), :] = torch.tensor(vector, dtype=torch_dtype)
    
    return padded_tensor


def compute_relevance_scores(query_embeddings, document_embeddings, k):
    """
    Compute relevance scores for top-k documents given a query.
    
    :param query_embeddings: Tensor representing the query embeddings, shape: [num_query_terms, embedding_dim]
    :param document_embeddings: Tensor representing embeddings for k documents, shape: [k, max_doc_length, embedding_dim]
    :param k: Number of top documents to re-rank
    :return: Sorted document indices based on their relevance scores
    """
    
    # Compute batch dot-product of Eq (query embeddings) and D (document embeddings)
    # Resulting shape: [k, num_query_terms, max_doc_length]
    scores = torch.matmul(query_embeddings.unsqueeze(0), document_embeddings.transpose(1, 2))
    
    # Apply max-pooling across document terms (dim=2) to find the max similarity per query term
    # Shape after max-pool: [k, num_query_terms]
    max_scores_per_query_term = scores.max(dim=2).values
    
    # Sum the scores across query terms to get the total score for each document
    # Shape after sum: [k]
    total_scores = max_scores_per_query_term.sum(dim=1)
    
    # Sort the documents based on their total scores
    sorted_indices = total_scores.argsort(descending=True)
    
    return sorted_indices, total_scores


if __name__ == "__main__":
    # Load a model from the HuggingFace hub (in this case the M2V_base_output model)
    model_name = "./m2v_model"
    model = StaticModel.from_pretrained(model_name)

    # Make sequences of token embeddings
    token_embeddings = model.encode_as_sequence(["It's a secret to everybody how they have fun.", "It's dangerous to go alone!", "cowabunga"])
    document_tensor = pad_and_stack_vectors(token_embeddings)
    print(f"Input shapes: {[v.shape for v in token_embeddings]}")
    print(f"Output shape: {document_tensor.shape}")

    # Compute relevance scores for a query
    query_embeddings = torch.tensor(model.encode_as_sequence("what is risky?"))
    relevance_scores = compute_relevance_scores(query_embeddings, document_tensor, k=len(token_embeddings))
    print("Sorted document indices based on relevance scores:", relevance_scores)