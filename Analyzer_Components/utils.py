from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN

import numpy as np


######################################################################################################################
##################################################### Embedder #######################################################
######################################################################################################################

class Embbeder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def vectorize(self, texts):
        return np.array(self.model.encode(texts, convert_to_numpy=True))

    def calculate_pairwise_cosine_similarity(self, vectors):
        return cosine_similarity(vectors)

    def vectorize_and_calculate_similarity(self, texts):
        vectors = self.vectorize(texts)
        return self.calculate_pairwise_cosine_similarity(vectors)


###############################################################################################################################
##################################################### Semantic Clusters #######################################################
###############################################################################################################################

class SemanticClusters:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        
        self.embedder = Embbeder(model_name)

    def compute_semantic_diversity_score(self, filtered_prompt, eps=0.5, min_samples=2):
        """
        Compute the Semantic Diversity Score (SDS) for a single filtered prompt.

        Parameters:
        filtered_prompt (list of str): The filtered prompt (list of words) to evaluate.
        eps (float): Maximum distance between two samples for DBSCAN clustering. Default is 0.5.
        min_samples (int): Minimum samples required to form a dense region for DBSCAN. Default is 2.

        Returns:
        float: The Semantic Diversity Score (SDS) for the filtered prompt.
        """
        if not filtered_prompt:
            return 0.0  # No words in the prompt

        # Generate embeddings for each word
        embeddings = self.embedder.vectorize(filtered_prompt)

        # Apply DBSCAN clustering
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
        labels = dbscan.fit_predict(embeddings)

        # Calculate number of clusters and noise ratio
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # Exclude noise
        noise_ratio = np.sum(labels == -1) / len(labels)  # Proportion of noise points

        # Compute SDS
        sds = num_clusters * (1 - noise_ratio)
        return round(sds, 2)

    def compute_semantic_repetition_penalty(self, filtered_prompt):
        """
        Compute the Semantic Repetition Penalty (SRP) for a single filtered prompt.

        Parameters:
        filtered_prompt (list of str): The filtered prompt (list of words) to evaluate.

        Returns:
        float: The Semantic Repetition Penalty (SRP) for the filtered prompt.
        """
        if len(filtered_prompt) < 2:
            return 1.0  # No repetition penalty for fewer than 2 words

        # Generate embeddings for the words
        embeddings = self.embedder.vectorize(filtered_prompt)

        # Calculate pairwise cosine similarity
        similarity_matrix = cosine_similarity(embeddings)

        # Compute the average similarity (excluding diagonal)
        n = len(similarity_matrix)
        upper_triangle_indices = np.triu_indices(n, k=1)
        avg_similarity = similarity_matrix[upper_triangle_indices].mean()

        # Compute SRP
        srp = 1 + avg_similarity
        return round(srp, 2)