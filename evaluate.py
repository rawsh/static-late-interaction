from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
from datasets import load_dataset
from model2vec import StaticModel
from tqdm import tqdm

from inference import PROJ_DIM, compute_relevance_scores, pad_and_stack_vectors


def load_and_prepare_data():
    """Load and prepare the scifact dataset"""
    print("Loading datasets...")
    correct_chunks = load_dataset("mteb/scifact", split="test")
    corpus = load_dataset("mteb/scifact", name="corpus", split="corpus")
    queries = load_dataset("mteb/scifact", name="queries", split="queries")
    
    # Format corpus
    corpus = corpus.map(lambda row: {
        'formatted': f"{row['title']} {row['text']}"
    })
    corpus_texts = corpus['formatted']
    
    # Create mapping dictionaries
    query_id_to_corpus_ids = defaultdict(list)
    for chunk in correct_chunks:
        query_id = chunk['query-id']
        corpus_id = chunk['corpus-id']
        query_id_to_corpus_ids[query_id].append(corpus_id)
    
    doc_index_to_chunk_id = {idx: chunk['_id'] for idx, chunk in enumerate(corpus)}
    query_id_to_question = {query['_id']: query['text'] for query in queries}
    
    # Validate mappings
    test_cases = list(query_id_to_corpus_ids.keys())
    query_ids = list(query_id_to_question.keys())
    assert all(query_id in query_ids for query_id in test_cases)
    
    return (corpus_texts, query_id_to_corpus_ids, doc_index_to_chunk_id, 
            query_id_to_question, test_cases)

def calculate_ndcg(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k: int = None
) -> float:
    """
    Calculate NDCG@k.
    For binary relevance (0 or 1), DCG = sum(rel_i / log2(i + 1)) where rel_i is 1 if doc is relevant
    IDCG is DCG of perfect ranking (all relevant docs first)
    """
    if not relevant_ids:
        return 0.0
        
    if k is not None:
        retrieved_ids = retrieved_ids[:k]
    
    # Calculate DCG
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_ids:
            dcg += 1.0 / np.log2(i + 2)  # i + 2 because 1-based ranking
            
    # Calculate IDCG
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant_ids), len(retrieved_ids))))
    
    if idcg == 0:
        return 0.0
        
    return dcg / idcg

def evaluate_model(
    model_path: str,
    batch_size: int = 32,
    show_progress: bool = True,
    device: str = None
):
    """Evaluate model on the scifact dataset."""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    
    # Load model and data
    model = StaticModel.from_pretrained(model_path)
    (corpus_texts, query_id_to_corpus_ids, doc_index_to_chunk_id,
     query_id_to_question, test_cases) = load_and_prepare_data()
    
    print(f"Number of test cases: {len(test_cases)}")
    print(f"Corpus size: {len(corpus_texts)}")
    
    # Encode corpus with batching
    print("Encoding corpus...")
    corpus_embeddings = []
    for i in tqdm(range(0, len(corpus_texts), batch_size)):
        batch_texts = corpus_texts[i:i + batch_size]
        batch_emb = model.encode_as_sequence(batch_texts)
        corpus_embeddings.extend(batch_emb)
        assert all(emb.shape[-1] == PROJ_DIM for emb in batch_emb)
    
    print("Creating padded tensor...")
    corpus_tensor, corpus_mask = pad_and_stack_vectors(
        corpus_embeddings,
        device=device,
        normalize=True
    )
    
    print(f"Corpus tensor shape: {corpus_tensor.shape}")
    
    # Encode queries
    print("Encoding queries...")
    queries = [query_id_to_question[qid] for qid in test_cases]
    query_embeddings = []
    for i in tqdm(range(0, len(queries), batch_size)):
        batch_queries = queries[i:i + batch_size]
        batch_emb = model.encode_as_sequence(batch_queries)
        query_embeddings.extend(batch_emb)
        assert all(emb.shape[-1] == PROJ_DIM for emb in batch_emb)
    
    # Initialize metrics
    metrics = {
        'recall@10': [],
        'recall@100': [],
        'recall@1000': [],
        'ndcg@10': [],
        'ndcg@100': [],
        'ndcg@1000': []
    }
    
    # For running averages
    running_avgs = {metric: 0.0 for metric in metrics.keys()}
    
    print("\nEvaluating queries...")
    pbar = tqdm(enumerate(test_cases), total=len(test_cases))
    for idx, query_id in pbar:
        # Get ground truth
        relevant_corpus_ids = query_id_to_corpus_ids[query_id]
        
        # Get query embeddings and compute scores
        query = torch.tensor(query_embeddings[idx], device=device)
        sorted_indices, _ = compute_relevance_scores(
            query,
            corpus_tensor,
            corpus_mask,
            batch_size=batch_size,
            device=device
        )
        
        # Convert indices to corpus IDs
        sorted_indices = sorted_indices.cpu().tolist()
        retrieved_ids = [doc_index_to_chunk_id[idx] for idx in sorted_indices]
        
        # Calculate metrics
        for k in [10, 100, 1000]:
            # Recall@K
            hits_k = set(retrieved_ids[:k]).intersection(relevant_corpus_ids)
            recall_k = len(hits_k) / len(relevant_corpus_ids) if relevant_corpus_ids else 0
            metrics[f'recall@{k}'].append(recall_k)
            
            # NDCG@K
            ndcg_k = calculate_ndcg(retrieved_ids, relevant_corpus_ids, k=k)
            metrics[f'ndcg@{k}'].append(ndcg_k)
        
        # Update running averages
        for metric in metrics.keys():
            running_avgs[metric] = (running_avgs[metric] * idx + metrics[metric][-1]) / (idx + 1)
        
        # Update progress bar with key metrics only
        pbar.set_postfix({
            'R@10': f"{running_avgs['recall@10']:.4f}",
            'NDCG@10': f"{running_avgs['ndcg@10']:.4f}"
        })
    
    # Print final results for all metrics
    print("\nFinal Results:")
    avg_metrics = {key: np.mean(values) for key, values in metrics.items()}
    for metric, value in avg_metrics.items():
        print(f"Average {metric}: {value:.4f}")
    
    return metrics, avg_metrics

if __name__ == "__main__":
    model_path = "./m2v_model"
    metrics, avg_metrics = evaluate_model(
        model_path=model_path,
        batch_size=32,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )