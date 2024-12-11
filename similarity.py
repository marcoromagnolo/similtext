from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import vstack
import joblib
import io
import os
from settings import STORAGE_DIR


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
    

    def get_document_scores(self, id):
        i = 0
        for page_id in self.page_ids:
            if page_id == id:
                tfidf_scores = self.tfidf_matrix[i].toarray()[0]
                feature_names = self.vectorizer.get_feature_names_out()
                return {feature_names[i]: tfidf_scores[i] for i in range(len(feature_names)) if tfidf_scores[i] > 0}
            i += 1
    

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
            if not documents:
                self.logger.error("No valid documents available for vectorization.")
                return
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

        return most_similar_document_id
    

    def add_document(self, document_id, document_text):
        new_tfidf_matrix = self.vectorizer.transform([document_text])

        # Save the vectorizer
        self.save_object("vectorizer", self.vectorizer)

        # Save the new tfidf_matrix
        self.tfidf_matrix = vstack([self.tfidf_matrix, new_tfidf_matrix])
        self.save_object("tfidf_matrix", self.tfidf_matrix)

        # Save the new page_ids
        self.page_ids.append(document_id)
        self.save_object("page_ids", self.page_ids)

        return True
