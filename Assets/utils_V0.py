from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA


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





    def apply_pca(self, embeddings, n_components=None):
            """
            Apply PCA to reduce the dimensionality of embeddings.
            
            Parameters:
            embeddings (numpy.ndarray): Word embeddings (n_samples x n_features).
            n_components (int or None): Number of components for PCA. If None, defaults to min(n_samples, n_features).
            
            Returns:
            numpy.ndarray: Transformed embeddings after PCA.
            """
            n_samples, n_features = embeddings.shape
            if n_components is None or n_components > min(n_samples, n_features):
                n_components = min(n_samples, n_features)  # Automatically adjust to a valid number of components
                print(f"Adjusting PCA components to {n_components} based on data size.")

            pca = PCA(n_components=n_components)
            reduced_embeddings = pca.fit_transform(embeddings)
            return reduced_embeddings


    def compute_semantic_diversity_score(self, filtered_prompt, n_components=5):
        """
        Compute SDS using PCA and entropy of reduced embeddings.
        
        Parameters:
        filtered_prompt (list of str): The filtered prompt (list of words).
        n_components (int or None): Number of components for PCA. If None, auto-adjusts to min(samples, features).
        
        Returns:
        float: Semantic Diversity Score (SDS).
        """
        if len(filtered_prompt) < 2:
            return 0.0  # Not enough words for meaningful computation

        # Generate embeddings
        embeddings = self.embedder.vectorize(filtered_prompt)

        # Apply PCA with automatic adjustment
        reduced_embeddings = self.apply_pca(embeddings, n_components=n_components)

        # Normalize reduced embeddings to probabilities (row-wise softmax)
        probabilities = np.exp(reduced_embeddings) / np.exp(reduced_embeddings).sum(axis=1, keepdims=True)

        # Compute entropy for each reduced embedding
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-8), axis=1)

        # Average entropy as SDS
        sds = np.mean(entropy)
        return sds

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
        return srp