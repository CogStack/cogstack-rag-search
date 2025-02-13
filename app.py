import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from openai import OpenAI
from dotenv import load_dotenv
import os
import sqlite3

nlp = spacy.load("en_core_web_md")

def create_embeddings(df):
    question_embs = [nlp(str(q)).vector for q in df["Question"]]
    answer_embs   = [nlp(str(a)).vector for a in df["Answer"]]
    return question_embs, answer_embs

def keyword_search(df, keywords):

    #If no keywords provided, return the entire DataFrame
    if not keywords:
        return df

    #Combine keywords into one regex pattern, separated by the OR symbol '|'
    pattern = "|".join(keywords)

    #Return rows where the answer column contains at least one keyword
    filtered_data = df[df["Answer"].str.contains(pattern, case=False, na=False)]

    return filtered_data

def semantic_search(df, question_embs, answer_embs, query, top_k=5):
    query_emb = nlp(query).vector

    qn_scores = cosine_similarity([query_emb], question_embs)[0]
    top_qn_idx = np.argsort(-qn_scores)[:top_k]

    return df.iloc[top_qn_idx][["ID","Question","Answer"]]

def run_query(table_name) -> pd.DataFrame:
    """
    Execute a SQL query against a local SQLite database table and return the results.
    """
    db_path = "mock_data//uclh_data_sources//mock_dbo.sqlite"

    #Connect to the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    #Query a specific table
    query = f"SELECT * FROM {table_name} LIMIT 5;"
    df = pd.read_sql(query, conn)

    #Close connection
    conn.close()

    #Show the data
    return df

def main():
    st.set_page_config(layout="wide")

    if "keywords" not in st.session_state:
        st.session_state["keywords"] = []
    if "keyword_input" not in st.session_state:
        st.session_state["keyword_input"] = ""

    #Callback function to add the current keyword and clear the input
    def add_keyword():
        user_input = st.session_state["keyword_input"].strip()
        if user_input:
            st.session_state["keywords"].append(user_input)
            #Clear the text input field
            st.session_state["keyword_input"] = ""

    #Load data
    df = pd.read_csv('qa.csv')

    #Load the environment variables from the .env file
    load_dotenv()
    key = os.environ.get("OPENAI_API_KEY")

    # Initialize OpenAI client
    client = OpenAI(api_key=key)

    # Initialize Streamlit session state variables
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-4o-mini"

    #Title
    st.title("CogStack Search Tool Mockup")

    st.write(
        "This is a mockup of an app for clinicians to search through EHRs / CogStack data using a mixture of traditional keyword search and semantic search."
        " It uses a synthetic Q&A dataset on Pituitary Adenoma made up of 100 Q&A pairs that has been created by prompting ChatGPT."
        " The underlying search functionality is very basic and the main purpose of this mockup is to help elicit requirements from users."
    )

    #Create four columns
    col1_left, col1_right = st.columns(2, gap="large")
    col2_left, col2_right = st.columns(2, gap="large")
    col3_left, col3_right = st.columns(2, gap="large")

    # ----------------------
    # LEFT COLUMN: KEYWORDS
    # ----------------------
    with col1_left:
        st.subheader("Keyword Filtering")
        st.write(
            "Here the user can enter keywords and add them to a list."
            " This list of keywords is used to pre-filter the Answers from the Q&A dataset."
            " The search will filter out any Q&A pairs where the Answer doesn't contain at least one of the added words."
        )
        #A small horizontal layout for the input + Add button
        kw_input_col, kw_add_btn_col = st.columns([3, 1], gap="small")

        with kw_input_col:
            st.text_input(
                label="Enter a single keyword",
                placeholder="E.g. 'pituitary'",
                key="keyword_input",
                label_visibility="collapsed"
            )

        with kw_add_btn_col:
            #The button uses the callback to add/clear the keyword
            st.button("Add", on_click=add_keyword)

        #Display the list of keywords
        st.write("Current Keywords:")
        st.write(st.session_state["keywords"])

    # ---------------------
    # RIGHT COLUMN: SEMANTIC SEARCH
    # ---------------------
    with col1_right:
        st.subheader("Semantic Search")
        st.write(
            "Here the user can enter a free-text query."
            " This will be used to do a semantic similarity search against the remaining Questions in the Q&A dataset."
            " The most similar Q&A pairs are retrieved and the Answers are provided to the LLM along with the original query."
        )
        free_text_query = st.text_area(
            label="",
            placeholder="Type your query here...",
            height=150
        )

        #Placeholder for LLM response
        #llm_response_placeholder = st.empty()

        if st.button("Search"):
            #If user hasn't entered anything at all
            if not free_text_query.strip() and not st.session_state["keywords"]:
                st.warning("Please enter a query or a keyword.")
            else:
                #1)Filter by keywords
                filtered_data = keyword_search(df, st.session_state["keywords"])

                #2)Further refine by semantic search
                question_embs, answer_embs = create_embeddings(filtered_data)
                top_results = semantic_search(filtered_data, question_embs, answer_embs, query=free_text_query.strip(), top_k=5)

                #Show the matches in the left column
                with col2_left:
                    st.subheader("Top 5 Semantic Matches")
                    st.write("The most similar Q&A pairs (based on the similarity between the queries) from the filtered data are shown below:")
                    st.write(top_results)
                    #st.write(top_results["Answer"].tolist())               
                    #for _, row in top_results.iterrows():
                        #st.markdown(f"**Q:** {row['Question']}\n\n**A:** {row['Answer']}")

                #Create a single prompt with the search results and user’s query
                results_prompt = "\n".join(f"- {ans}" for ans in top_results["Answer"].tolist())
                
                prompt = f"""
                You are a helpful assistant. Below are some extracted passages relevant to the user's query:

                {results_prompt}

                The user's query is: "{free_text_query.strip()}"

                Based on these passages and the query, please provide a concise response:
                """

                #Call the OpenAI API with this prompt
                completion = client.chat.completions.create(
                    model=st.session_state["openai_model"],
                    messages=[
                        {"role": "system", "content": prompt}
                    ]
                )

                #Show the LLM response in the right column
                with col2_right:
                    st.subheader("LLM Response")
                    st.write("This is the response produced by the LLM after passing it the original query and the Answers from the top 5 matches:")
                    st.write(completion.choices[0].message.content)

    with col3_left:
        st.subheader("Structured Data Query")
        st.write("Enter the **name** of a database table to view data from it. A list of table names is provided on the right hand side")

        table_name = st.text_input("Table Name", "PatientDim")

        if st.button("Run DB Query"):
            try:
                results_df = run_query(table_name)
                st.write("Results:")
                st.dataframe(results_df)
            except Exception as e:
                st.error(f"Error running query: {e}")

    with col3_right:
        db_path = "mock_data//uclh_data_sources//mock_dbo.sqlite"

        #Connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)
        st.write("Tables in database:", tables)

        #Close connection
        conn.close()
        
if __name__ == "__main__":
    main()