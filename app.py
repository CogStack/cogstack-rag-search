import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from openai import OpenAI
from dotenv import load_dotenv
import os

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

    #Use str.contains with case=False (ignore case) and na=False (exclude NaN)
    filtered_data = df[df["Answer"].str.contains(pattern, case=False, na=False)]

    return filtered_data

def semantic_search(df, question_embs, answer_embs, query, top_k=5):
    query_emb = nlp(query).vector

    qn_scores = cosine_similarity([query_emb], question_embs)[0]
    top_qn_idx = np.argsort(-qn_scores)[:top_k]

    return df.iloc[top_qn_idx][["ID","Question","Answer"]]

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
    st.title("CogStack Search Mockup - Pituitary Adenoma")

    #Create two columns
    col_left, col_right = st.columns(2, gap="large")

    # ----------------------
    # RIGHT COLUMN: KEYWORDS
    # ----------------------
    with col_right:
        st.subheader("Keyword Search")
        st.write(
            "Enter single-word keywords below to filter the data."
            " Press Add to include each keyword in the filter list."
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

        #Placeholder for LLM response
        llm_response_placeholder = st.empty()

    # ---------------------
    # LEFT COLUMN: FREE TEXT
    # ---------------------
    with col_left:
        st.subheader("Free Text (Semantic) Search")
        st.write(
            "Enter a free-text query below."
            " This will be used to do a semantic search and retrieve the most relevant text from the database."
        )
        free_text_query = st.text_area(
            label="",
            placeholder="Type your query here...",
            height=150
        )

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
                
                #Display results
                st.subheader("Top 5 Semantic Matches")
                #st.write(top_results)
                st.write(top_results["Answer"].tolist())
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
                with col_right:
                    st.subheader("LLM Response")
                    st.write(completion.choices[0].message.content)
                     
if __name__ == "__main__":
    main()