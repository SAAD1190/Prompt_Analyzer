##################################################################################################################
################################################## Dependencies ##################################################
##################################################################################################################

import json
import string
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, cmudict
from nltk.tree import Tree
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util
from nltk.translate.bleu_score import sentence_bleu
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




##################################################################################################################
############################################## Class Initialization ##############################################
##################################################################################################################

class PromptAnalyzer:
    def __init__(self, prompts_list, model_name="all-mpnet-base-v2"):
        """
        Initialize the PromptAnalyzer with a list of prompts.
        Parameters:
        prompts_list (list): A list of prompts (strings).
        model_name (str): Name of the embedding model to use. Default is 'all-MiniLM-L6-v2'.
        """
        self.prompts_list = prompts_list
        self.vectorizer = TfidfVectorizer()
        self.semantic_clusters = SemanticClusters(model_name)
        self.embedding_model = SentenceTransformer(model_name)

##################################################################################################################
################################################## Main Analyzer #################################################
##################################################################################################################


    def process_prompts(prompts, sort_by="svr", reference_prompts=None, reverse=True, output_file="processed_prompts.json"):
        """
        Process prompts and return sorted results based on the specified metric, then save the results to a JSON file.

        Parameters:
        prompts (list): List of prompts to analyze.
        sort_by (str): Metric to sort by. Options are:
                    "vr" (Vocabulary Richness),
                    "sr" (Semantic Richness),
                    "svr" (Semantic Vocabulary Richness),
                    "relevance" (requires reference_prompts),
                    "lexical_density",
                    "parse_tree_depth".
        reference_prompts (list): A list of reference prompts for the relevance metric. Default is None.
        reverse (bool): Whether to sort in descending order. Default is True.
        output_file (str): Name of the JSON file to write the results. Default is 'processed_prompts.json'.

        Returns:
        list: A list of tuples (prompt, score) sorted by the specified metric.
        """
        analyzer = PromptAnalyzer(prompts)

        if sort_by == "vr":
            results = analyzer.compute_vocabulary_richness()
        elif sort_by == "sr":
            results = analyzer.compute_semantic_richness()
        elif sort_by == "svr":
            results = analyzer.compute_svr()
        elif sort_by == "lexical_density":
            lexical_densities = analyzer.lexical_density()
            results = [(prompt, score) for prompt, score in zip(prompts, lexical_densities)]
        elif sort_by == "parse_tree_depth":
            parse_tree_depths = analyzer.parse_tree_depth()
            results = [(prompt, depth) for prompt, depth in zip(prompts, parse_tree_depths)]
        elif sort_by == "relevance":
            if reference_prompts is None:
                raise ValueError("Reference prompts must be provided for the relevance metric.")
            results = analyzer.relevance(reference_prompts)
        else:
            raise ValueError(f"Invalid sort_by value: {sort_by}. Choose from 'vr', 'sr', 'svr', 'relevance', 'lexical_density', 'parse_tree_depth'.")

        # Sort the results by score
        sorted_results = sorted(results, key=lambda x: x[1], reverse=reverse)

        # Write results to JSON using PromptAnalyzer's method
        analyzer.write_to_json(sorted_results, filename=output_file)

        print(f"Results saved to '{output_file}'")
        return sorted_results


    ##################################################################################################################
    ########################################### Initial Prompts Processing ###########################################
    ##################################################################################################################

    def preprocess_prompts(self):
        """
        Process prompts by removing punctuation, tokenizing, and filtering out stop words.
        Returns:
        tuple: A tuple containing the following lists:
            - prompts_unpunctuated (list of str): Prompts with punctuation removed.
            - prompts_filtered (list of list of str): Tokenized and filtered prompts.
            - prompts_length (list of int): Length of each filtered prompt.
            - unique_words_list (list of set of str): Unique words in each filtered prompt.
        """
        stop_words = set(stopwords.words('english'))
        punct_table = str.maketrans('', '', string.punctuation)
        prompts_unpunctuated = []
        prompts_filtered = []
        prompts_length = []
        unique_words_list = []

        for prompt in self.prompts_list:
            prompt_unpunctuated = prompt.translate(punct_table)
            words = word_tokenize(prompt_unpunctuated)
            prompt_filtered = [word for word in words if word.lower() not in stop_words]
            prompt_length = len(prompt_filtered)
            unique_words = set(prompt_filtered)

            prompts_unpunctuated.append(prompt_unpunctuated)
            prompts_filtered.append(prompt_filtered)
            prompts_length.append(prompt_length)
            unique_words_list.append(unique_words)

        return prompts_unpunctuated, prompts_filtered, prompts_length, unique_words_list

##################################################################################################################
############################################### Semantic Metrics #################################################
##################################################################################################################

# ------- Vocabulary Richness ------- #

    def compute_vocabulary_richness(self):
        """
        Calculate the vocabulary richness score for each prompt based on the ratio of unique words to total words.
        Returns:
        list: A list of vocabulary richness scores for all prompts (not sorted).
        """
        _, prompts_filtered, _, _ = self.preprocess_prompts()
        vocabulary_richness_scores = []

        for filtered_prompt in prompts_filtered:
            prompt_length = len(filtered_prompt)
            unique_words_number = set(filtered_prompt)
            vocabulary_richness = len(unique_words_number) / prompt_length if prompt_length > 0 else 0
            vocabulary_richness_scores.append(round(vocabulary_richness, 2))

        return vocabulary_richness_scores

# ------- Semantic Richness ------- #

    def compute_semantic_richness(self):
        """
        Compute the Semantic Richness (SR) for all prompts.

        Returns:
        list: A list of Semantic Richness (SR) scores for all prompts (not sorted).
        """
        # _, prompts_filtered, _, _ = self.prompt_processing()
        semantic_richness_scores = []

        for prompt in self.prompts_list:
            sds = self.semantic_clusters.compute_semantic_diversity_score(prompt)
            srp = self.semantic_clusters.compute_semantic_repetition_penalty(prompt)
            sr = sds / srp if srp > 0 else 0  # Avoid division by zero
            semantic_richness_scores.append(round(sr, 3))

        return semantic_richness_scores

# ------- Semantic Vocabulary Richness ------- #

    def compute_svr(self):
        """
        Compute the Semantic Vocabulary Richness (SVR) for all prompts.

        SVR is calculated as:
            SVR = SR (Semantic Richness) x VR (Vocabulary Richness)

        Parameters:
        eps (float): Maximum distance between two samples for DBSCAN clustering. Default is 0.7.
        min_samples (int): Minimum samples required to form a dense region for DBSCAN. Default is 2.

        Returns:
        list: A list of Semantic Vocabulary Richness (SVR) scores for all prompts (not sorted).
        """
        vr_scores = self.compute_vocabulary_richness()
        sr_scores = self.compute_semantic_richness()
        svr_scores = [round(sr * vr, 3) for sr, vr in zip(sr_scores, vr_scores)]
        return svr_scores

    ##################################################################################################################
    ################################################### Utilities ####################################################
    ##################################################################################################################

    def write_to_json(self, results, filename="prompt_results.json"):
        """
        Write results to a JSON file.

        Parameters:
        results (dict or list): Results to save.
        filename (str): Name of the JSON file. Default is 'prompt_results.json'.
        """
        with open(filename, 'w', encoding='utf-8') as json_file:
            json.dump(results, json_file, indent=4, ensure_ascii=False)
        print(f"Results saved to '{filename}'")

    ##################################################################################################################
    ################################################### Metrics ######################################################
    ##################################################################################################################

    def lexical_density(self):
        """
        Calculate the lexical density for each prompt.

        Returns:
        list: A list of lexical density scores for all prompts (not sorted).
        """
        _, prompts_filtered, _, _ = self.preprocess_prompts()
        content_words = {'NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS'}
        lexical_densities = []

        for filtered_prompt in prompts_filtered:
            tagged_words = nltk.pos_tag(filtered_prompt)
            content_word_count = sum(1 for word, tag in tagged_words if tag in content_words)
            lexical_density = content_word_count / len(filtered_prompt) if len(filtered_prompt) > 0 else 0
            lexical_densities.append(round(lexical_density, 2))

        return lexical_densities


    ##################################################################################################################
    ################################################ Syntax Complexity ################################################
    ##################################################################################################################

    def parse_tree_depth(self):
        """
        This function calculates the average parse tree depth for each prompt.

        Returns:
        list: A list of average parse tree depths for all prompts (not sorted).
        """
        def get_parse_tree(sentence):
            """
            Generate a parse tree for a sentence using a predefined grammar.
            """
            tokens = nltk.word_tokenize(sentence)
            tagged = nltk.pos_tag(tokens)
            grammar = """
                NP: {<DT>?<JJ>*<NN>}
                PP: {<IN><NP>}
                VP: {<VB.*><NP|PP|CLAUSE>+$}
                CLAUSE: {<NP><VP>}
            """
            cp = nltk.RegexpParser(grammar)
            try:
                tree = cp.parse(tagged)
                return tree
            except:
                return None

        def tree_depth(tree):
            """
            Recursively calculate the depth of a parse tree.
            """
            if isinstance(tree, Tree):
                return 1 + max(tree_depth(child) for child in tree) if tree else 0
            else:
                return 0

        prompt_depths = []
        for prompt in self.prompts_list:
            depths = []
            sentences = sent_tokenize(prompt)
            for sentence in sentences:
                parse_tree = get_parse_tree(sentence)
                if parse_tree:
                    depth = tree_depth(parse_tree)
                    depths.append(depth)
            avg_depth = sum(depths) / len(depths) if depths else 0
            prompt_depths.append(round(avg_depth, 2))

        return prompt_depths

    ##################################################################################################################
    ################################################ Relevance Metrics ################################################
    ##################################################################################################################

    # def relevance(self, reference_prompts):
    #     """
    #     This function calculates the relevance of each prompt in `self.prompts_list` compared to a set of reference prompts.

    #     Parameters:
    #     reference_prompts (list): A list of reference prompts to compare against.

    #     Returns:
    #     list: A list of relevance scores for all prompts (not sorted).
    #     """
    #     vectorizer = TfidfVectorizer()
    #     relevance_scores = []

    #     for prompt in self.prompts_list:
    #         tfidf_matrix = vectorizer.fit_transform([prompt] + reference_prompts)
    #         similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
    #         relevance_score = similarity_matrix.mean()
    #         relevance_scores.append(round(relevance_score, 2))

    #     return relevance_scores

    def relevance(self, reference_prompts, ls=0.4, ss=0.4, sts=0.2): 
        """
        Compute hybrid relevance scores by comparing each prompt in prompts_list 
        to its corresponding reference in reference_prompts.

        Parameters:
        reference_prompts (list): A list of reference prompts (same length as prompts_list).
        ls (float): Weight for lexical similarity (TF-IDF).
        ss (float): Weight for semantic similarity (all-mpnet-base-v2).
        threshold (float): Upper threshold for flagging potential non-paraphrases.

        Returns:
        list: A list of combined relevance scores for each prompt-reference pair.
        """
        if len(self.prompts_list) != len(reference_prompts):
            raise ValueError("The prompts list and reference prompts list must have the same length.")
        
        bleu_weights = (0.2, 0.5, 0.2, 0.1)  # BLEU focuses on unigram and bigram precision
        combined_scores = []

        for prompt, reference in zip(self.prompts_list, reference_prompts):
            # TF-IDF Lexical Similarity
            tfidf_matrix = self.vectorizer.fit_transform([prompt, reference])
            tfidf_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).mean()
            # Normalize TF-IDF score
            tfidf_score = min(tfidf_score, 1.0)
            lexical_score = tfidf_score

            # all-mpnet-base-v2 Semantic Similarity
            embeddings = self.embedding_model.encode([prompt, reference], convert_to_tensor=True)
            semantic_score = util.cos_sim(embeddings[0], embeddings[1]).item()
            # Normalize Semantic Similarity
            semantic_score = min(semantic_score, 1.0)

            # Structural Similarity (BLEU Score)
            bleu_score = sentence_bleu([reference.split()], prompt.split(), weights=bleu_weights)
            bleu_score = min(bleu_score, 1.0)  # Normalize BLEU to [0, 1]
            stuctural_score = bleu_score


            # Combine scores (weighted average)
            combined_score = (ls * lexical_score) + (ss * semantic_score) + (sts * stuctural_score)

            # Apply a manual threshold
            combined_score = round(combined_score, 3)
            combined_scores.append(combined_score)

        return combined_scores

    def remove_redundancy(self, threshold=0.85):
        """
        Remove redundant prompts from self.prompts_list while keeping the most relevant ones.

        Parameters:
        threshold (float): Redundancy threshold for cosine similarity.

        Returns:
        list: A list of unique, most relevant prompts.
        """
        # Step 1: Compute embeddings for all prompts
        embeddings = self.embedding_model.encode(self.prompts_list, convert_to_tensor=True)
        
        # Step 2: Compute pairwise relevance scores
        relevance_scores = self.relevance(self.prompts_list)  # Scores of prompts with respect to themselves
        
        keep_indices = []
        redundant_indices = set()
        n = len(self.prompts_list)

        # Step 3: Compare pairwise similarities
        for i in range(n):
            if i in redundant_indices:
                continue
            keep_indices.append(i)

            for j in range(i + 1, n):
                if j not in redundant_indices:
                    # Compute pairwise similarity (cosine similarity)
                    similarity = util.cos_sim(embeddings[i], embeddings[j]).item()

                    if similarity > threshold:
                        # Keep the more relevant prompt
                        if relevance_scores[i] >= relevance_scores[j]:
                            redundant_indices.add(j)
                        else:
                            redundant_indices.add(i)
                            keep_indices.remove(i)
                            break  # Stop processing 'i' as it is marked redundant

        # Step 4: Collect unique prompts
        unique_prompts = [self.prompts_list[i] for i in keep_indices]
        return unique_prompts
