from model2vec.distill import distill
from transformers import AutoTokenizer

# Choose a Sentence Transformer model
# model_name = "BAAI/bge-base-en-v1.5"
model_name = "answerdotai/answerai-colbert-small-v1"

# # Load a pre-trained tokenizer (e.g., BERT)
# # tokenizer = AutoTokenizer.from_pretrained("./m2v_bert")
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# # Get the vocabulary as a dictionary
# vocab_dict = tokenizer.get_vocab()

# # Convert the dictionary to a list of words
# vocab_list = list(vocab_dict.keys())
# qry_list = [f"[unused0]{token}" for token in vocab_list]
# doc_list = [f"[unused1]{token}" for token in vocab_list]
# combined = [item for pair in zip(qry_list, doc_list) for item in pair] + qry_list[len(doc_list):] + doc_list[len(qry_list):]

# # Print the first 10 words in the vocabulary
# print(combined[:10])

# # Print the total number of words in the vocabulary
# print(f"Total vocabulary size: {len(combined)}")

# # Distill the model
# m2v_model = distill(model_name=model_name, pca_dims=128, vocabulary=combined, use_subword=False)
m2v_model = distill(model_name=model_name, pca_dims=128)

# # Save the model
m2v_model.save_pretrained("m2v_model")
