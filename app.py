import streamlit as st
from prompt_analyzer import PromptAnalyzer  # Replace with the correct module name
import pandas as pd  # For creating and displaying tables

# Streamlit app title
st.title("Prompt Analyzer")

# Sidebar for user inputs
st.sidebar.header("Analyze Your Prompts")

# Side note about first run delay
st.sidebar.markdown(
    """
    **Note:** The first run might take some time as models and embedders are downloaded locally. 
    Subsequent runs will be faster.
    """
)

# Input prompts (multi-line text box for multiple prompts)
user_prompts = st.sidebar.text_area(
    "Enter test prompts (one per line as shown below):",
    placeholder="Example:\nThis is the first test prompt.\nHere is the second test prompt.",
)

# Select sorting metric
sort_by = st.sidebar.selectbox(
    "Sort by:",
    [
        "Semantic Vocabulary Richness",
        "Semantic Richness",
        "Vocabulary Richness",
        "Lexical Density",
        "Parse Tree Depth",
        "Relevance",  # Added Relevance option
    ],
    index=0,
)

# Allow reference prompt entry if "Relevance" is selected
reference_prompts_list = None
if sort_by == "Relevance":
    reference_prompts = st.sidebar.text_area(
        "Enter reference prompts (one per line, matching the test prompts order):",
        placeholder="Example:\nThis is the reference for the first test prompt.\nThis is the reference for the second test prompt.",
    )
    if reference_prompts.strip():
        reference_prompts_list = reference_prompts.strip().split("\n")

# Allow file download for processed results
enable_download = st.sidebar.checkbox("Enable file download for results")

# Analyze button
if st.sidebar.button("Analyze"):
    if user_prompts.strip():
        # Split test prompts into a list
        prompts_list = user_prompts.strip().split("\n")

        # Initialize the PromptAnalyzer with the user's test prompts
        analyzer = PromptAnalyzer(prompts_list)

        # Process prompts based on the selected sorting metric
        if sort_by == "Relevance":
            # Ensure reference prompts are provided and match in count
            if not reference_prompts_list:
                st.error("Please enter reference prompts to compute relevance.")
                st.stop()
            elif len(prompts_list) != len(reference_prompts_list):
                st.error("The number of test prompts and reference prompts must match!")
                st.stop()

            # Use relevance metric
            sorted_results = analyzer.process_prompts(
                sort_by="Relevance",
                reference_prompts=reference_prompts_list,
                output_file="processed_prompts.json",
            )
        else:
            # For non-relevance metrics
            sorted_results = analyzer.process_prompts(
                sort_by=sort_by,
                output_file="processed_prompts.json",
            )

        # Display the results in a table
        st.subheader("Sorted Prompts and Scores")
        # Convert sorted_results to a Pandas DataFrame for tabular display
        results_df = pd.DataFrame(sorted_results, columns=["Prompt", "Score"])
        st.table(results_df)  # Display the table in Streamlit

        # Enable downloading results as a JSON file
        if enable_download:
            with open("processed_prompts.json", "r") as file:
                json_data = file.read()
            st.download_button(
                label="Download Results as JSON",
                data=json_data,
                file_name="processed_prompts.json",
                mime="application/json",
            )
    else:
        st.error("Please enter at least one test prompt to analyze!")