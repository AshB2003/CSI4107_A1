import json
import math
import re
import string
from collections import defaultdict, Counter
from nltk.stem.snowball import EnglishStemmer
from nltk.tokenize import TreebankWordTokenizer

def preprocess_query(text, stopwords):
    """
    Preprocess the query text to match the document preprocessor:
      - Remove URLs.
      - Tokenize using TreebankWordTokenizer.
      - Convert tokens to lowercase.
      - Remove punctuation and stopwords.
      - Discard tokens containing any digits.
      - Stem tokens using the EnglishStemmer.
    """
    # Remove URLs
    text = re.sub(r"http\S+|https\S+|www\S+", "", text)
    
    tokenizer = TreebankWordTokenizer()
    stemmer = EnglishStemmer()
    tokens = tokenizer.tokenize(text)
    processed_tokens = []
    
    for token in tokens:
        # Lowercase and strip punctuation from the ends
        token = token.lower().strip(string.punctuation)
        if not token or token in stopwords:
            continue
        # Remove tokens containing digits
        if re.search(r'\d', token):
            continue
        processed_tokens.append(stemmer.stem(token))
    
    return processed_tokens

def build_query_vector(tokens, total_docs, inverted_index):
    """
    Build a query vector with weights computed as:
        weight = (TF in query) * (IDF)
    where IDF is computed as log(total_docs / df) for tokens that occur in the index.
    Only tokens that appear in the inverted index are considered.
    """
    tf_counts = Counter(tokens)
    query_vector = {}
    for token, tf in tf_counts.items():
        if token in inverted_index:
            df = inverted_index[token]["df"]
            # Compute IDF using the same formula as indexing
            idf = math.log(total_docs / df)
            query_vector[token] = tf * idf
    return query_vector

def compute_query_norm(query_vector):
    """Compute the Euclidean norm of the query vector."""
    return math.sqrt(sum(weight**2 for weight in query_vector.values()))

def retrieve_documents(query_vector, query_norm, inverted_index, doc_vector_lengths):
    """
    For each token in the query vector, look up its posting list in the inverted index
    and accumulate the dot-product contributions from documents.
    Then, convert the accumulated dot product to a cosine similarity using the document norm.
    
    Returns:
        A dictionary mapping document IDs to their cosine similarity scores.
    """
    doc_scores = defaultdict(float)
    
    # For each token in the query vector, add contributions from each document
    for token, q_weight in query_vector.items():
        if token in inverted_index:
            postings = inverted_index[token]["postings"]
            for doc_id, doc_weight in postings.items():
                doc_scores[doc_id] += q_weight * doc_weight
    
    # Compute cosine similarity by dividing by the product of norms
    cosine_similarities = {}
    for doc_id, dot_product in doc_scores.items():
        doc_norm = doc_vector_lengths.get(doc_id, 0)
        if doc_norm > 0 and query_norm > 0:
            cosine_similarities[doc_id] = dot_product / (query_norm * doc_norm)
    return cosine_similarities

def main():
    run_name = "run"
    
    # File paths (adjusted to your folder structure)
    stopwords_file = "scifact/stopwords.txt"
    inverted_index_file = "scifact/inverted_index.json"
    doc_vector_lengths_file = "scifact/document_vector_lengths.json"
    queries_file = "scifact/queries.jsonl"
    results_file = "scifact/results.txt"
    
    # Load stopwords
    with open(stopwords_file, "r", encoding="utf-8") as f:
        stopwords = set(line.strip() for line in f)
    
    # Load the inverted index (weighted index) produced by indexing.py
    with open(inverted_index_file, "r", encoding="utf-8") as f:
        inverted_index = json.load(f)
    
    # Load document vector lengths (norms)
    with open(doc_vector_lengths_file, "r", encoding="utf-8") as f:
        doc_vector_lengths = json.load(f)
    
    # Total number of documents (needed for query IDF computation)
    total_docs = len(doc_vector_lengths)
    
    # Load queries (each line is a JSON object)
    queries = []
    with open(queries_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    
    # Open results file for writing
    with open(results_file, "w", encoding="utf-8") as out_f:
        for query_obj in queries:
            # Assuming each query has fields "_id" and "text"
            qid = query_obj["_id"]
            q_text = query_obj["text"]
            
            # Preprocess the query text
            tokens = preprocess_query(q_text, stopwords)
            
            # Build the query vector using TF * IDF
            query_vector = build_query_vector(tokens, total_docs, inverted_index)
            query_norm = compute_query_norm(query_vector)
            
            # Retrieve documents and compute cosine similarity scores
            cosine_similarities = retrieve_documents(query_vector, query_norm, inverted_index, doc_vector_lengths)
            
            # Sort documents by descending similarity score
            ranked_docs = sorted(cosine_similarities.items(), key=lambda x: x[1], reverse=True)
            
            # Write top-100 results in the required format:
            # query_id Q0 doc_id rank score run_name
            for rank, (doc_id, score) in enumerate(ranked_docs[:100], start=1):
                out_f.write(f"{qid} Q0 {doc_id} {rank} {score} {run_name}\n")
    
    print("Retrieval complete. Results saved to", results_file)

if __name__ == "__main__":
    main()
