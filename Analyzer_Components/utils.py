from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import numpy as np
from scipy.spatial import ConvexHull

##########################################################################################################################
##################################################### Embedder ###########################################################
##########################################################################################################################

class Embbeder:
    def __init__(self, model_name="all-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)

    def vectorize_chunks(self, chunks):
        """Generate embeddings for a list of chunks."""
        return np.array(self.model.encode(chunks, convert_to_numpy=True))
    
    def chunker(self,text, chunk_size=5, overlap=2):
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

    
    # def compute_semantic_diversity_score(self, text, chunk_size=5, overlap=2):
    #     # Divide prompt into chunks
    #     chunks = self.embedder.chunker(text, chunk_size, overlap)

    #     # Generate embeddings for each chunk
    #     embeddings = self.embedder.vectorize_chunks(chunks)

    #     # Normalize reduced embeddings row-wise to probabilities
    #     probabilities = np.exp(embeddings) / (np.exp(embeddings).sum(axis=1, keepdims=True) + 1e-10)

    #     # Compute entropy for each chunk
    #     entropy = -np.sum(probabilities * np.log(probabilities + 1e-10), axis=1)

    #     # Average entropy across chunks
    #     sds = np.mean(entropy)
    #     return sds




    def compute_semantic_diversity_score(self, text, chunk_size=5, overlap=2):
        # Step 1: Divide prompt into chunks
        chunks = self.embedder.chunker(text, chunk_size, overlap)

        # Step 2: Generate embeddings for each chunk
        embeddings = self.embedder.vectorize_chunks(chunks)

        # Step 3: Ensure embeddings have enough points for ConvexHull
        if len(embeddings) < embeddings.shape[1] + 1:
            # ConvexHull requires at least (n+1) points for n-dimensional embeddings
            return 0.0  # Low diversity if insufficient points

        # Step 4: Compute the convex hull of the embeddings
        hull = ConvexHull(embeddings)

        # Step 5: Calculate the volume of the convex hull
        diversity_score = hull.volume

        return diversity_score


    def compute_semantic_repetition_penalty(self, text, chunk_size=5, overlap=2):
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
    
##################################################################################################################
# Experimenting embedding handlings


    # def compute_semantic_diversity_score(self, text, chunk_size=5, overlap=2):
    #         """
    #         Compute Semantic Diversity Score (SDS) using cosine similarity spread.

    #         Parameters:
    #         - text (str): The input text.
    #         - chunk_size (int): Number of words in each chunk.
    #         - overlap (int): Number of overlapping words between consecutive chunks.

    #         Returns:
    #         - float: Semantic Diversity Score (SDS) based on cosine similarity spread.
    #         """
    #         # Divide the text into overlapping chunks
    #         chunks = self.embedder.chunker(text, chunk_size, overlap)

    #         # Generate embeddings for the chunks
    #         embeddings = self.embedder.vectorize_chunks(chunks)

    #         # Compute pairwise cosine similarity
    #         similarity_matrix = cosine_similarity(embeddings)

    #         # Extract the upper triangle (excluding diagonal) of the similarity matrix
    #         upper_triangle_indices = np.triu_indices(len(similarity_matrix), k=1)
    #         similarities = similarity_matrix[upper_triangle_indices]

    #         # Compute the spread of cosine similarities
    #         sds = np.mean(similarities)

    #         return sds