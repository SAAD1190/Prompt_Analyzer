from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np


######################################################################################################################
##################################################### Embedder #######################################################
######################################################################################################################

class Embbeder:
    def __init__(self, model_name="all-mpnet-base-v2"):
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
    def __init__(self, model_name="all-mpnet-base-v2"):
        
        self.embedder = Embbeder(model_name)

    def compute_semantic_diversity_score(self, filtered_prompt):
        """
        Compute SDS using entropy of word embeddings.

        Parameters:
        filtered_prompt (list of str): The filtered prompt (list of words).

        Returns:
        float: Semantic Diversity Score (SDS).
        """
        if len(filtered_prompt) < 2:
            return 0.0  # Not enough words for entropy computation

        embeddings = self.embedder.vectorize(filtered_prompt)

        # Normalize embeddings to probabilities (row-wise softmax)
        probabilities = np.exp(embeddings) / np.exp(embeddings).sum(axis=1, keepdims=True)

        # Compute entropy for each word
        entropy = -np.sum(probabilities * np.log(probabilities), axis=1)

        # Average entropy across the prompt
        sds = np.mean(entropy)
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