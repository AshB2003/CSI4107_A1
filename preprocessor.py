import json
import re
import string
from collections import Counter
from nltk.stem.snowball import EnglishStemmer
from nltk.tokenize import TreebankWordTokenizer

def do_preprocessor(corpus_path, stopwords_path):
    # Create a set of punctuation characters
    punct_set = set(string.punctuation)
    
    # Load stopwords from the stopwords.txt file
    with open(stopwords_path, "r", encoding="utf-8") as f:
        stopwords = set(line.strip() for line in f)
    
    # Dictionary to hold the processed tokens for each document
    document_word_dict = {}
    
    # Open the corpus.jsonl file and process it line by line
    with open(corpus_path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # Skip empty lines
            doc = json.loads(line)
            
            # Get the document ID
            doc_id = doc["_id"]
            
            # Combine title and text (if title exists) into one text string
            text = ""
            if "title" in doc:
                text += doc["title"] + " "
            if "text" in doc:
                text += doc["text"]
            
            # Remove links
            text = re.sub(r"http\S+", "", text)
            text = re.sub(r"https\S+", "", text)
            text = re.sub(r"www\S+", "", text)
            
            # Initialize list for tokens
            words = []
            tokenizer = TreebankWordTokenizer()
            
            # Tokenize and process each token
            for token in tokenizer.tokenize(text):
                token_lower = token.lower()
                # Only consider tokens that are not stopwords or punctuation
                if token_lower not in stopwords and token_lower not in punct_set:
                    try:
                        # Apply stemming
                        stemmed = EnglishStemmer().stem(token_lower)
                        words.append(stemmed)
                    except Exception:
                        words.append(token_lower)
            
            # Remove any punctuation that might remain (using a translation table)
            table = str.maketrans('', '', string.punctuation)
            words = [w.translate(table) for w in words]
            
            # Remove any tokens that might have become stopwords after translation
            words = [w for w in words if w not in stopwords]
            
            # Remove tokens that contain numbers (keep only purely alphabetic words)
            words = [w for w in words if re.match(r'^[a-zA-Z]+$', w)]
            
            # Additional pre-processing: remove specific unwanted unicode characters
            words = [w.replace(u"\u201d", "") for w in words]
            
            # Store the processed tokens in the dictionary. If possible, convert the id to int.
            try:
                key = int(doc_id)
            except ValueError:
                key = doc_id
            document_word_dict[key] = words

    # Save the processed document-word dictionary to the scifact folder
    output_path = "scifact/document_word_dict.json"
    with open(output_path, "w", encoding="utf-8") as out_file:
        json.dump(document_word_dict, out_file, indent=4)
    
    # Create a dictionary that maps each document id to a Counter of its word frequencies
    document_word_count_dict = {}
    for doc in document_word_dict:
        document_word_count_dict[doc] = Counter(document_word_dict[doc])
    
    return document_word_dict, document_word_count_dict

if __name__ == "__main__":
    # Define paths
    corpus_path = "scifact/corpus.jsonl"
    stopwords_path = "scifact/stopwords.txt"
    
    # Run the preprocessor
    doc_dict, doc_count_dict = do_preprocessor(corpus_path, stopwords_path)
    print("Preprocessing complete. Processed", len(doc_dict), "documents.")
