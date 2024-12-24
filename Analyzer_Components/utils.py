from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import numpy as np
from sklearn.metrics.pairwise import cosine_distances

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


#------- Compute Semantic Diversity Score (SDS) using cosine similarity spread -------#

    def compute_semantic_diversity_score(self, text, chunk_size=5, overlap=2):
        """
        Compute the semantic diversity score for a given text based on cosine distances.
        
        Args:
            text (str): Input text or prompt.
            chunk_size (int): Number of words per chunk.
            overlap (int): Number of overlapping words between consecutive chunks.
            normalize (bool): Whether to normalize the score to the range [0, 1].
        Returns:
            float: Diversity score.
        """
        # Step 1: Divide prompt into chunks
        chunks = self.embedder.chunker(text, chunk_size, overlap)
        # Step 2: Generate embeddings for each chunk
        embeddings = self.embedder.vectorize_chunks(chunks)
        # Step 3: Normalize embeddings
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Step 4: Compute pairwise cosine distances
        distances = cosine_distances(embeddings)
        # Step 5: Calculate the average distance
        avg_distance = np.mean(distances)
        diversity_score = avg_distance

        return diversity_score



#------- Compute Semantic Repetition Penalty (SRP) for a prompt divided into overlapping chunks -------#

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