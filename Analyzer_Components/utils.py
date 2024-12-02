from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import numpy as np


##########################################################################################################################
##################################################### Embedder ###########################################################
##########################################################################################################################

class Embbeder:
    def __init__(self, model_name="all-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)

    def vectorize_chunks(self, chunks):
        """Generate embeddings for a list of chunks."""
        return np.array(self.model.encode(chunks, convert_to_numpy=True))
    
    def chunker(text, chunk_size=5, overlap=2):
        words = text.split()
        chunks = []
        step = chunk_size - overlap  # Step size to move the sliding window
        
        # Create chunks using a sliding window
        for i in range(0, len(words), step):
            chunk = words[i:i + chunk_size]
            chunks.append(" ".join(chunk))
        
        return chunks


##############################################################################################################################
##################################################### Chunk-Based Semantic Clusters #########################################
##############################################################################################################################

class SemanticClusters:
    def __init__(self, model_name="all-mpnet-base-v2"):
        self.embedder = Embbeder(model_name)

    
    def compute_chunked_semantic_diversity_score(self, text, chunk_size=5, overlap=2):
        # Divide prompt into chunks
        chunks = self.embedder.chunker(text, chunk_size, overlap)

        # Generate embeddings for each chunk
        embeddings = self.embedder.vectorize_chunks(chunks)

        # Normalize reduced embeddings row-wise to probabilities
        probabilities = np.exp(embeddings) / (np.exp(embeddings).sum(axis=1, keepdims=True) + 1e-10)

        # Compute entropy for each chunk
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10), axis=1)

        # Average entropy across chunks
        sds = np.mean(entropy)
        return sds

    def compute_chunked_semantic_repetition_penalty(self, text, chunk_size=5, overlap=2):
        """
        Compute SRP for a prompt divided into overlapping chunks.

        Parameters:
        text (str): The input prompt.
        chunk_size (int): Number of words per chunk.
        overlap (int): Number of overlapping words between consecutive chunks.

        Returns:
        float: Semantic Repetition Penalty (SRP) for the prompt.
        """
        # Divide prompt into chunks
        chunks = self.embedder.chunker(text, chunk_size, overlap)

        # Generate embeddings for each chunk
        embeddings = self.embedder.vectorize_chunks(chunks)

        # Calculate pairwise cosine similarity
        similarity_matrix = cosine_similarity(embeddings)

        # Compute the average similarity (excluding diagonal)
        n = len(similarity_matrix)
        upper_triangle_indices = np.triu_indices(n, k=1)
        avg_similarity = similarity_matrix[upper_triangle_indices].mean()

        # Compute SRP
        srp = 1 + avg_similarity
        return srp
    
# def compute_chunked_semantic_diversity_score(self, text, chunk_size=5, overlap=2, n_components=5):
#         """
#         Compute SDS for a prompt divided into overlapping chunks.

#         Parameters:
#         text (str): The input prompt.
#         chunk_size (int): Number of words per chunk.
#         overlap (int): Number of overlapping words between consecutive chunks.
#         n_components (int): Number of principal components to keep. Default is 5.

#         Returns:
#         float: Semantic Diversity Score (SDS) for the prompt.
#         """
#         # Divide prompt into chunks
#         chunks = self.embedder.chunker(text, chunk_size, overlap)

#         # Generate embeddings for each chunk
#         embeddings = self.embedder.vectorize_chunks(chunks)

#         # Apply PCA to reduce dimensionality
#         pca = PCA(n_components=n_components)
#         reduced_embeddings = pca.fit_transform(embeddings)

#         # Normalize reduced embeddings row-wise to probabilities
#         probabilities = np.exp(reduced_embeddings) / (np.exp(reduced_embeddings).sum(axis=1, keepdims=True) + 1e-10)

#         # Compute entropy for each chunk
#         entropy = -np.sum(probabilities * np.log(probabilities + 1e-10), axis=1)

#         # Average entropy across chunks
#         sds = np.mean(entropy)
#         return sds