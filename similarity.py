from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import vstack
import joblib
import io
import os
import requests
import json
import db
from settings import STORAGE_DIR, OPENAI_SECRET


class Similarity:

    def __init__(self, logger):
        self.logger = logger
        self.vectorizer = None
        self.storage_dir = STORAGE_DIR
        # Ensure the storage directory exists
        os.makedirs(self.storage_dir, exist_ok=True)
        self.vectorizer = self.load_object("vectorizer")
        self.tfidf_matrix = self.load_object("tfidf_matrix")
        self.page_ids = self.load_object("page_ids")
        if not self.vectorizer:
            self.build_vectorizer()


    def get_document_ids(self):
        return self.page_ids
    

    def serialize_object(self, obj):
        """Serialize an object to binary using joblib."""
        buffer = io.BytesIO()
        joblib.dump(obj, buffer)
        buffer.seek(0)
        return buffer.read()


    def deserialize_object(self, binary_data):
        """Deserialize an object from binary using joblib."""
        buffer = io.BytesIO(binary_data)
        return joblib.load(buffer)


    def save_object(self, object_name, obj):
        """Save a serialized object to a file."""
        file_path = os.path.join(self.storage_dir, f"{object_name}.pkl")
        with open(file_path, 'wb') as f:
            joblib.dump(obj, f)


    def load_object(self, object_name):
        """Load a serialized object from a file."""
        file_path = os.path.join(self.storage_dir, f"{object_name}.pkl")
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                return joblib.load(f)
        return None


    def build_vectorizer(self, data=None):
        """Retrieve a document for the given page ID."""

        self.vectorizer = TfidfVectorizer()
        self.page_ids = []
        self.tfidf_matrix = None
        
        # update data
        if data:
            self.page_ids = [row[0] for row in data if row[0] is not None]
            documents = [row[1] for row in data if row[1] is not None]
            self.tfidf_matrix = self.vectorizer.fit_transform(documents)

        # Save
        self.save_object("vectorizer", self.vectorizer)
        self.save_object("tfidf_matrix", self.tfidf_matrix)
        self.save_object("page_ids", self.page_ids)


    def get_most_similar_document_id(self, document_id, document_text):
        """Update the similarity for a specific page."""

        if not document_id:
            print(f"Empty id document")
            return

        if not document_text:
            print(f"Empty text document")
            return
        
        new_tfidf_matrix = self.vectorizer.transform([document_text])

        # Compute cosine similarity with all existing documents
        similarity = cosine_similarity(new_tfidf_matrix, self.tfidf_matrix)

        # Get the max similarity score and its index
        max_similarity_score = similarity.max().item()
        most_similar_document_id = None
        if max_similarity_score > 0:
            max_similarity_index = similarity.argmax()

            # Find the corresponding page ID using the index
            most_similar_document_id = self.page_ids[max_similarity_index] if self.page_ids else None
            print(f"This document is most similar to id:{most_similar_document_id} with a similarity of {max_similarity_score}")

        # Save the vectorizer
        self.save_object("vectorizer", self.vectorizer)

        # Save the new tfidf_matrix
        self.tfidf_matrix = vstack([self.tfidf_matrix, new_tfidf_matrix])
        self.save_object("tfidf_matrix", self.tfidf_matrix)

        # Save the new page_ids
        self.page_ids.append(document_id)
        self.save_object("page_ids", self.page_ids)

        return most_similar_document_id


    def compare_texts(self, text1, text2):

            # Create headers
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENAI_SECRET}"
                }

            # Create body
            data = {
                "model": 'gpt-3.5-turbo',
                "messages": [
                        {"role": "system", "content": "You are a professional journalist evaluating the similarity of two articles. Two articles are considered similar if they discuss the same topic, cite the same sources or protagonists, and present the same informations."},
                        {'role': 'user', 'content': f"This is the first article: {text1}."},
                        {'role': 'user', 'content': f"This is the second article: {text2}."},
                        {'role': 'user', 'content': f"Reply with 'true' if articles are similar otherwise reply with 'false'."},
                    ],
                "temperature": 0,
                "response_format": {"type": "text"}
            }

            if self.call_openapi(headers, data):
                data = {
                    "model": 'gpt-3.5-turbo',
                    "messages": [
                            {"role": "system", "content": "You are a professional journalist and you dont want to repeat the same informations in two articles."},
                            {'role': 'user', 'content': f"This is the first article: {text1}."},
                            {'role': 'user', 'content': f"This is the second article: {text2}."},
                            {'role': 'user', 'content': "Reply with 'true' if the two articles have contextual overlap by discussing the same main subject and providing the same information and details. Otherwise, reply with 'false'."}
                        ],
                    "temperature": 0,
                    "response_format": {"type": "text"}
                }
                return self.call_openapi(headers, data)
            
            return False    

    def call_openapi(self, headers, data):
        self.logger.info(f"Request: {str(data)}")
        url = "https://api.openai.com/v1/chat/completions"
        response = requests.post(url, headers=headers, json=data)

        if response.status_code == 200:
            result = response.json()  # Decodifica la risposta in formato JSON
            out = result["choices"][0]["message"]["content"]
            self.logger.info(f"Response: {out}")
            return out in {"true", "True", "TRUE", "1", "yes", "y"}

        else:
            self.logger.info(f"Errore: {response.status_code}")
            self.logger.info(response.text)
            
        return False    