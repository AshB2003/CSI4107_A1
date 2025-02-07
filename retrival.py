import json
import math
import re
import string
from collections import Counter, defaultdict
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem.snowball import EnglishStemmer

def preprocess_text(text, stopwords):
    """
    Preprocess text in the same way as document processing:
      - Remove URLs.
      - Tokenize using TreebankWordTokenizer.
      - Convert tokens to lowercase.
      - Remove tokens that are stopwords or are solely punctuation.
      - Stem tokens using EnglishStemmer.
      - Remove any residual punctuation using a translation table.
      - Remove tokens that become stopwords after punctuation removal.
      - Remove tokens that contain any numbers (only keep purely alphabetic tokens).
      - Remove specific unwanted unicode characters (e.g., \u201d).
      
    Returns:
        A list of processed tokens.
    """
    # Remove URLs (http, https, and www links)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    
    # Initialize the tokenizer and stemmer
    tokenizer = TreebankWordTokenizer()
    stemmer = EnglishStemmer()
    
    # Tokenize the text
    tokens = tokenizer.tokenize(text)
    words = []
    punct_set = set(string.punctuation)
    
    # Process each token: lowercase, check stopwords/punctuation, and apply stemming
    for token in tokens:
        token_lower = token.lower()
        if token_lower in stopwords or token_lower in punct_set:
            continue
        try:
            stemmed = stemmer.stem(token_lower)
            words.append(stemmed)
        except Exception:
            words.append(token_lower)
    
    # Remove any residual punctuation from tokens using a translation table
    table = str.maketrans('', '', string.punctuation)
    words = [w.translate(table) for w in words]
    
    # Remove tokens that might have become stopwords after punctuation removal
    words = [w for w in words if w not in stopwords]
    
    # Remove tokens that contain numbers (only keep purely alphabetic tokens)
    words = [w for w in words if re.match(r'^[a-zA-Z]+$', w)]
    
    # Remove specific unwanted unicode characters (e.g., \u201d)
    words = [w.replace(u"\u201d", "") for w in words]
    
    return words

def build_query_vector(tokens, weighted_dict, vocab_size):
    """
    Compute a query vector using a normalized term frequency times idf.
    Only tokens that exist in the weighted_dict (i.e. in the corpus vocabulary) are used.
    
    The weight formula used here is:
       weight = (0.5 + 0.5*(tf / max_tf)) * idf
    where idf = log(vocab_size / doc_freq, 2) and doc_freq is the number
    of documents containing the term.
    """
    token_counts = Counter(tokens)
    if not token_counts:
        return {}
    max_tf = max(token_counts.values())
    q_vector = {}
    for term, tf in token_counts.items():
        if term in weighted_dict:
            # In our weighted_dict, postings are stored as {doc_id: weight}
            # But for the purpose of IDF calculation, we need the document frequency.
            # We assume that the length of weighted_dict[term] gives the doc_freq.
            doc_freq = len(weighted_dict[term])
            idf = math.log(float(vocab_size) / doc_freq, 2)
            q_vector[term] = (0.5 + 0.5 * (tf / float(max_tf))) * idf
    return q_vector

def compute_document_norms(weighted_dict):
    """
    Precompute and return the Euclidean norm of each document vector.
    The weighted_dict is a mapping: term -> {doc_id: weight}.
    For each document, we sum the squared term weights (over all terms) and take the square root.
    """
    doc_norms = defaultdict(float)
    for term, postings in weighted_dict.items():
        for doc_id, weight in postings.items():
            doc_norms[doc_id] += weight ** 2
    # Convert to Euclidean norm by taking the square root
    for doc_id in doc_norms:
        doc_norms[doc_id] = math.sqrt(doc_norms[doc_id])
    return doc_norms

def retrieve_documents(query_vector, weighted_dict):
    """
    For each term in the query vector, add contributions to documents that contain the term.
    Returns a dictionary mapping document IDs to the accumulated dot product between
    the query and the document vectors.
    """
    scores = defaultdict(float)
    for term, q_weight in query_vector.items():
        # Only process terms that appear in the weighted index.
        if term in weighted_dict:
            for doc_id, d_weight in weighted_dict[term].items():
                scores[doc_id] += q_weight * d_weight
    return scores

def main():
    run_name = "myRun"
    # File paths
    stopwords_file = "scifact/stopwords.txt"
    weighted_dict_file = "scifact/weighted_dict.json"
    queries_file = "scifact/queries.jsonl"
    results_file = "scifact/results.txt"
    query_processed_file = "scifact/query_processed.json"
    
    # Load stopwords
    with open(stopwords_file, "r", encoding="utf-8") as f:
        stopwords = set(line.strip() for line in f)
    
    # Load the weighted dictionary (the inverted index with tf-idf weights)
    with open(weighted_dict_file, "r", encoding="utf-8") as f:
        weighted_dict = json.load(f)
    vocab_size = len(weighted_dict)
    
    # Precompute document norms for cosine similarity computations
    doc_norms = compute_document_norms(weighted_dict)
    
    # Load queries from queries.jsonl (each line is a JSON object)
    queries = []
    with open(queries_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    
    # Create a dictionary to store the processed queries
    processed_queries = {}
    
    # Process each query and write ranked results
    with open(results_file, "w", encoding="utf-8") as out_f:
        for query_obj in queries:
            # Extract query id and text.
            # Here, we assume each query JSON has fields "_id" and "text".
            qid = query_obj["_id"]
            q_text = query_obj["text"]
            
            # Preprocess the query text using the unified function
            tokens = preprocess_text(q_text, stopwords)
            # Store the processed tokens in our dictionary
            processed_queries[qid] = tokens
            
            query_vector = build_query_vector(tokens, weighted_dict, vocab_size)
            if not query_vector:
                continue  # Skip if the query has no valid tokens
            
            # Compute the norm of the query vector
            q_norm = math.sqrt(sum(weight ** 2 for weight in query_vector.values()))
            
            # Retrieve documents and accumulate dot product scores
            raw_scores = retrieve_documents(query_vector, weighted_dict)
            
            # Compute cosine similarity scores
            cosine_scores = {}
            for doc_id, dot_product in raw_scores.items():
                if q_norm > 0 and doc_norms.get(doc_id, 0) > 0:
                    cosine_scores[doc_id] = dot_product / (q_norm * doc_norms[doc_id])
                else:
                    cosine_scores[doc_id] = 0.0
            
            # Rank the documents by descending similarity score
            ranked_docs = sorted(cosine_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Write the top-100 results to the output file in the required format:
            # query_id Q0 doc_id rank score run_name
            for rank, (doc_id, score) in enumerate(ranked_docs[:100], start=1):
                out_f.write(f"{qid} Q0 {doc_id} {rank} {score} {run_name}\n")
    
    # Write the processed queries to a file
    with open(query_processed_file, "w", encoding="utf-8") as qp_f:
        json.dump(processed_queries, qp_f, indent=4)
    
    print("Retrieval complete. Results saved to", results_file)
    print("Processed queries saved to", query_processed_file)

if __name__ == "__main__":
    main()
