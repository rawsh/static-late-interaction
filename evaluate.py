from model2vec import StaticModel
import torch
from datasets import load_dataset

from inference import pad_and_stack_vectors, compute_relevance_scores

# load the scifact dataset
correct_chunks = load_dataset("mteb/scifact", split="test")
corpus = load_dataset("mteb/scifact", name="corpus", split="corpus")
queries = load_dataset("mteb/scifact", name="queries", split="queries")
print(queries)

# load model2vec model
model_name = "./m2v_model"
model = StaticModel.from_pretrained(model_name)

# format corpus
def format_corpus(row):
    return f"{row['title']} {row['text']}"
corpus = corpus.map(lambda row: {'formatted': format_corpus(row)})
corpus_texts = corpus['formatted']
print(f"Number of corpus documents: {len(corpus_texts)}")

# encode corpus
corpus_emb = model.encode_as_sequence(corpus_texts)
print(f"Embedding dimension: {corpus_emb[0].shape[1]}")

# big tensor
corpus_tensor = pad_and_stack_vectors(corpus_emb)
print(f"Corpus tensor shape: {corpus_tensor.shape}")

# scoring dict
query_id_to_corpus_map = {}
for chunk in correct_chunks:
    query_id = chunk['query-id']
    corpus_id = chunk['corpus-id']
    if query_id not in query_id_to_corpus_map:
        query_id_to_corpus_map[query_id] = []
    query_id_to_corpus_map[query_id] += [corpus_id]

# question texts
query_id_to_question = {}
for query in queries:
    query_id_to_question[query['_id']] = query['text']

# all test cases
test_cases = query_id_to_corpus_map.keys()
query_ids = query_id_to_question.keys()

# make sure all test cases are in the query ids
assert all(query_id in query_ids for query_id in test_cases)
print("Number of test cases:", len(test_cases))

# encode queries
query_emb = model.encode_as_sequence(list(query_id_to_question.values()))
assert len(query_emb) == len(query_ids)
print(f"Number of queries: {len(query_ids)}")

# accuracy: number of hits in top k
accuracy_scores = []
for query_id, query_emb in zip(query_ids, query_emb):
    # get the relevant corpus ids ground truth
    relevant_corpus_ids = query_id_to_corpus_map[query_id]
    print(relevant_corpus_ids)
    # compute relevance scores
    query = torch.tensor(query_emb)
    scores = compute_relevance_scores(query, corpus_tensor, k=len(corpus_tensor))
    print(scores)
    # get the top k indices
    _, top_k_indices = torch.topk(scores, k=10)
    # convert to list
    top_k_indices = top_k_indices.tolist()
    # check if any of the top k indices are in the relevant corpus ids
    hits = [idx for idx in top_k_indices if idx in relevant_corpus_ids]
    # compute accuracy
    accuracy = len(hits) / len(relevant_corpus_ids)
    accuracy_scores.append(accuracy)
    # print the question and the top k results
    print(f"Question: {query_id_to_question[query_id]}")
    print(f"Top 10 results: {top_k_indices[:10]}")
    print(f"Relevant corpus ids: {relevant_corpus_ids}")
    print(f"Accuracy: {accuracy}")
    print("--------------------------------------------------")
