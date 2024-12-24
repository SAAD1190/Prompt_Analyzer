from Analyzer_Components.prompt_analyzer import PromptAnalyzer

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