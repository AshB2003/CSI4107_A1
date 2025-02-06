import json
import math

def build_inverted_index(document_word_dict):
    """
    Build a raw inverted index from the document_word_dict.
    
    For each token in the vocabulary, the index will contain:
      - A posting list: a dictionary mapping document IDs to the raw term frequency (TF)
      - The document frequency (DF): the number of documents in which the token appears
      
    Parameters:
        document_word_dict (dict): Mapping from document IDs to a list of preprocessed tokens.
        
    Returns:
        inverted_index (dict): Mapping from token to {"postings": {doc_id: TF, ...}, "df": DF}
    """
    inverted_index = {}
    for doc_id, tokens in document_word_dict.items():
        # Count frequencies in this document
        token_counts = {}
        for token in tokens:
            token_counts[token] = token_counts.get(token, 0) + 1
        
        # Update the inverted index with the token counts for this document
        for token, tf in token_counts.items():
            if token not in inverted_index:
                inverted_index[token] = {"postings": {}, "df": 0}
            # Add the term frequency for this document
            inverted_index[token]["postings"][doc_id] = tf
            # Increase the document frequency for this token
            inverted_index[token]["df"] += 1
    return inverted_index

def weight_index(inverted_index, total_docs):
    """
    Convert the raw inverted index to a weighted index.
    
    For each token, compute the inverse document frequency (IDF) using:
         IDF = log(total_docs / DF)
    Then, for each posting, compute the weight as:
         weight = TF * IDF
         
    The inverted_index is updated in place.
    
    Parameters:
        inverted_index (dict): The raw inverted index.
        total_docs (int): Total number of documents in the collection.
    """
    for token, data in inverted_index.items():
        df = data["df"]
        # Compute IDF; using natural logarithm (change base if desired)
        idf = math.log(total_docs / df)
        # Update each posting: weight = TF * IDF
        for doc_id, tf in data["postings"].items():
            data["postings"][doc_id] = tf * idf

def compute_document_vector_lengths(inverted_index):
    """
    Compute the Euclidean length of each document vector.
    
    For each document, the length is computed as:
         length(doc) = sqrt(sum_{token in doc} (weight)^2)
    
    Parameters:
        inverted_index (dict): The weighted inverted index.
        
    Returns:
        doc_vector_lengths (dict): Mapping from document ID to its vector length.
    """
    doc_vector_lengths = {}
    # Loop over each token's posting list
    for token, data in inverted_index.items():
        for doc_id, weight in data["postings"].items():
            doc_vector_lengths[doc_id] = doc_vector_lengths.get(doc_id, 0) + weight ** 2
    # Take square root of the sum of squares for each document
    for doc_id in doc_vector_lengths:
        doc_vector_lengths[doc_id] = math.sqrt(doc_vector_lengths[doc_id])
    return doc_vector_lengths

def do_indexer(document_word_dict):
    """
    Main indexing function.
    
    1. Build a raw inverted index from the document tokens.
    2. Compute the weighted index (TF * IDF) after knowing DF values.
    3. Compute the vector length for each document.
    
    Parameters:
        document_word_dict (dict): Preprocessed document tokens.
        
    Returns:
        inverted_index (dict): The weighted inverted index.
        doc_vector_lengths (dict): Mapping from document IDs to their vector lengths.
    """
    total_docs = len(document_word_dict)
    # Step 1: Build raw inverted index
    inverted_index = build_inverted_index(document_word_dict)
    
    # Step 2: Compute weights for each posting (TF * IDF)
    weight_index(inverted_index, total_docs)
    
    # Step 3: Compute document vector lengths (for use in cosine similarity)
    doc_vector_lengths = compute_document_vector_lengths(inverted_index)
    
    # Save the results for later use in retrieval
    with open("scifact/inverted_index.json", "w", encoding="utf-8") as f:
        json.dump(inverted_index, f, indent=4)
    with open("scifact/document_vector_lengths.json", "w", encoding="utf-8") as f:
        json.dump(doc_vector_lengths, f, indent=4)
    
    return inverted_index, doc_vector_lengths

if __name__ == "__main__":
    # Load preprocessed tokens from the document_word_dict file
    with open("scifact/document_word_dict.json", "r", encoding="utf-8") as f:
        document_word_dict = json.load(f)
    
    # Create the index and compute document vector lengths
    inverted_index, doc_vector_lengths = do_indexer(document_word_dict)
    print("Indexing complete.")
    print(f"Total documents: {len(document_word_dict)}")
    print(f"Vocabulary size: {len(inverted_index)}")
