import language_tool_python
import random
import csv
import json
import string
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, cmudict
from nltk.tree import Tree
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .utils import SemanticClusters
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util
from nltk.translate.bleu_score import sentence_bleu


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
    ########################################### Initial Prompts Processing ###########################################
    ##################################################################################################################

    def prompt_processing(self):
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

    def compute_vocabulary_richness(self):
        """
        Calculate the vocabulary richness score for each prompt.

        Returns:
        list: A list of vocabulary richness scores for all prompts (not sorted).
        """
        _, prompts_filtered, _, _ = self.prompt_processing()
        vocabulary_richness_scores = []

        for filtered_prompt in prompts_filtered:
            prompt_length = len(filtered_prompt)
            unique_words_number = set(filtered_prompt)
            vocabulary_richness = len(unique_words_number) / prompt_length if prompt_length > 0 else 0
            vocabulary_richness_scores.append(round(vocabulary_richness, 2))

        return vocabulary_richness_scores

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
        _, prompts_filtered, _, _ = self.prompt_processing()
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

    def relevance(self, reference_prompts, ls=0.4, ss=0.4):
        """
        Compute hybrid relevance scores by comparing each prompt in prompts_list 
        to its corresponding reference in reference_prompts.

        Parameters:
        reference_prompts (list): A list of reference prompts (same length as prompts_list).
        ls (float): Weight for lexical similarity (TF-IDF).
        ss (float): Weight for semantic similarity (all-mpnet-base-v2).
        sts (float): Weight for structural similarity (BLEU).

        Returns:
        list: A list of combined relevance scores for each prompt-reference pair.
        """
        if len(self.prompts_list) != len(reference_prompts):
            raise ValueError("The prompts list and reference prompts list must have the same length.")
        
        combined_scores = []

        for prompt, reference in zip(self.prompts_list, reference_prompts):
            # TF-IDF Lexical Similarity
            tfidf_matrix = self.vectorizer.fit_transform([prompt, reference])
            tfidf_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).mean()

            # all-mpnet-base-v2 Semantic Similarity
            embeddings = self.embedding_model.encode([prompt, reference], convert_to_tensor=True)
            semantic_score = util.cos_sim(embeddings[0], embeddings[1]).item()

            # # BLEU Structural Similarity
            # bleu_score = sentence_bleu([reference.split()], prompt.split(), weights=bleu_weights)

            # Combine scores (weighted average)
            combined_score = (ls * tfidf_score) + (ss * semantic_score)
            combined_scores.append(round(combined_score, 3))

        return combined_scores


    ##################################################################################################################
    ################################################ Readability Metrics ################################################
    ##################################################################################################################

    def prompt_readability(self):
        """
        Calculate the Flesch readability score for all prompts.

        Returns:
        list: A list of readability scores (Flesch scores) for all prompts (not sorted).
        """
        flesch_scores = []
        for prompt in self.prompts_list:
            flesch_score = self.readability(prompt)
            flesch_scores.append(flesch_score)
        return flesch_scores

    def readability(self, prompt):
        """
        Calculate the Flesch readability score for a single prompt.

        Parameters:
        prompt (str): A single prompt.

        Returns:
        float: The Flesch readability score for the prompt.
        """
        sentences = sent_tokenize(prompt)
        words = word_tokenize(prompt)
        num_sentences = len(sentences)
        num_words = len(words)
        d = cmudict.dict()

        def count_syllables(word):
            """
            Count syllables in a word using the CMU Pronouncing Dictionary.
            """
            pronunciation_list = d.get(word.lower())
            if not pronunciation_list:
                return 0
            pronunciation = pronunciation_list[0]
            return sum(1 for s in pronunciation if s[-1].isdigit())

        num_syllables = sum(count_syllables(word) for word in words)
        flesch_score = round(206.835 - 1.015 * (num_words / num_sentences) - 84.6 * (num_syllables / num_words), 2)
        return flesch_score
    

