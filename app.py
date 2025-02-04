import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from openai import OpenAI

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

    ans_scores = cosine_similarity([query_emb], answer_embs)[0]
    top_ans_idx = np.argsort(-ans_scores)[:top_k]

    qn_scores = cosine_similarity([query_emb], question_embs)[0]
    top_qn_idx = np.argsort(-qn_scores)[:top_k]

    final_idx = list(top_ans_idx) + list(top_qn_idx)
    return df.iloc[final_idx][["ID","Question","Answer"]]

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

    #Create or load embeddings
    question_embs, answer_embs = create_embeddings(df)

    # Initialize OpenAI client
    client = OpenAI(api_key=key)

    # Initialize Streamlit session state variables
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-4o-mini"

    # Initialize session state for background conversation history
    if 'background_messages' not in st.session_state:
        st.session_state.background_messages = [

        #Provide information about the task, the role the LLM should assume and the expected output
        {"role": "system", "content": 'Below is some background information about pituitary adenoma'},
        
    ]

    #Title
    st.title("CogStack Search Mockup - Pituitary Adenoma")

    #Create two columns
    col_left, col_right = st.columns([3, 2], gap="large")

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

                filtered_indices = filtered_data.index
                filtered_question_embs = [question_embs[i] for i in filtered_indices]
                filtered_answer_embs   = [answer_embs[i]   for i in filtered_indices]
                
                #2)Further refine by semantic search
                top_results = semantic_search(filtered_data, filtered_question_embs, filtered_answer_embs, query=free_text_query.strip(), top_k=5)
                
                #Display results
                st.subheader("Top 10 Semantic Matches")

                ##Option 1
                st.write(top_results)
                #st.write(top_results["Answer"].tolist())
                #for _, row in top_results.iterrows():
                    #st.markdown(f"**Q:** {row['Question']}\n\n**A:** {row['Answer']}")

                #Create prompt based on results
                results_prompt = "\n".join(f"- {text}" for text in top_results["Answer"].tolist())    

                #Add results and query to the background conversation history
                context = f"{results_prompt}\n\nUser Query: {free_text_query.strip()}"
                st.session_state.background_messages.append({"role": "system", "content": context})

                #Add a final prompt for the generation
                final_prompt = "Using the information above and the user's query, provide a response"
                st.session_state.background_messages.append({"role": "system", "content": final_prompt})

                #Combine background and visible messages for the API call
                combined_messages = st.session_state.background_messages

                #Print combined messages for debugging
                print("Combined Messages:")
                for msg in combined_messages:
                    print(f"Role: {msg['role']}, Content: {msg['content']}\n")

                #Generate aresponse from OpenAI
                completion = client.chat.completions.create(
                    model=st.session_state["openai_model"],
                    messages=[
                            {"role": m["role"], "content": m["content"]}
                            for m in combined_messages
                        ]
                )

                st.write(completion.choices[0].message)
                     

if __name__ == "__main__":
    main()