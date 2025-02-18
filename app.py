import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from openai import OpenAI
from dotenv import load_dotenv
import os
import sqlite3
import datetime

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
    #query = f"SELECT * FROM {table_name} LIMIT 5;"
    #df = pd.read_sql(query, conn)

    #table_name = "HospitalAdmissionFact"
    query = f"SELECT * FROM {table_name} LIMIT 5;"
    df = pd.read_sql(query, conn)

    #Close connection
    conn.close()

    #Show the data
    return df

def classify_query(query: str, client: OpenAI) -> str:
    """
    Use LLM to classify the query type as either 'medical_question', 'structured', or 'hybrid'
    """
    prompt = """Classify the following healthcare query into one of these categories:
    - medical_question: General medical questions or knowledge-based queries
    - structured: Queries about patient data, lab results, or records. This includes any queries mentioning amounts or numbers
    - hybrid: Queries that need both medical knowledge and patient data

    Query: {query}
    
    Return only one word: medical_question, structured, or hybrid"""

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt.format(query=query)}
        ]
    )
    return completion.choices[0].message.content.strip().lower()

def generate_medical_response(results: pd.DataFrame, query: str, client: OpenAI) -> str:
    """
    Generate a synthesized response from medical knowledge search results using LLM
    """
    # Combine all relevant answers into context
    context = "\n".join(results["Answer"].tolist())
    
    prompt = f"""Based on the following medical information, provide a clear and concise answer to the question.
    
    Question: {query}
    
    Relevant Information:
    {context}
    
    Please synthesize this information into a coherent response. If there isn't enough information to fully answer the question, acknowledge this in your response."""

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a medical knowledge assistant. Provide accurate, clear responses based on the given information."},
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content.strip()

def unified_search(query: str, df: pd.DataFrame, client: OpenAI):
    """
    Unified search pipeline that routes queries appropriately based on type
    """
    if not query.strip():
        return None, None, None, "Please enter a query."

    # Classify the query
    query_type = classify_query(query, client)
    
    text_results = None
    structured_results = None
    synthesized_response = None
    
    if query_type in ['medical_question', 'hybrid']:
        # Create embeddings for semantic search
        question_embs, answer_embs = create_embeddings(df)
        text_results = semantic_search(df, question_embs, answer_embs, query, top_k=5)
        # Generate synthesized response for medical questions
        synthesized_response = generate_medical_response(text_results, query, client)
    
    if query_type in ['structured', 'hybrid']:
        try:
            # For now, we'll just show relevant tables
            # In future, implement text2sql here
            db_path = "mock_data//uclh_data_sources//mock_dbo.sqlite"
            conn = sqlite3.connect(db_path)
            #structured_results = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)
            structured_results = run_query("HospitalAdmissionFact")
            conn.close()
        except Exception as e:
            st.error(f"Database error: {e}")
    
    return text_results, structured_results, synthesized_response, query_type

def main():
    st.set_page_config(layout="wide")
    
    # Load data and initialize OpenAI client
    df = pd.read_csv('qa.csv')
    load_dotenv()
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"

    # Move filters to sidebar
    with st.sidebar:
        st.title("Filters")
        
        # Patient ID filter
        st.text_input(
            "Patient ID",
            placeholder="Enter patient ID...",
            key="patient_id_filter"
        )
        
        # Keyword filter
        st.text_input(
            "Keywords",
            placeholder="Enter keywords...",
            key="keyword_filter"
        )
        
        # Date range filter with slider
        min_date = datetime.date(2000, 1, 1)
        max_date = datetime.date.today()
        
        dates = st.slider(
            "Select Date Range",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="DD/MM/YYYY"
        )

    # Main content now takes full width
    st.title("CogStack Search Tool")
    
    st.write(
        "This tool allows you to search through medical knowledge and patient data using natural language queries. "
        "Simply type your question, and the system will automatically determine whether to search medical knowledge, "
        "patient records, or both.\n\n"
        "You can also filter by patient ID, keywords, and date range."
    )

    # Single search interface
    query = st.text_area(
        "Enter your query",
        placeholder="e.g., 'What are the complications of pituitary surgery in diabetic patients?'",
        height=100
    )

    if st.button("Search"):
        if query:
            with st.spinner("Processing query..."):
                text_results, structured_results, synthesized_response, query_type = unified_search(query, df, client)
                
                st.write(f"Query classified as: **{query_type}**")
                
                # Show AI response
                if synthesized_response:
                    st.subheader("AI Response")
                    st.write(synthesized_response)
                
                # Show structured data results
                if structured_results is not None:
                    st.divider()
                    st.subheader("Relevant Patient Data")
                    st.dataframe(structured_results)
                    st.info("Note: Text-to-SQL functionality coming soon!")

if __name__ == "__main__":
    main()