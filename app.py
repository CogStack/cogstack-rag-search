import streamlit as st
import pandas as pd

df = pd.read_csv('qa.csv')

def keyword_search(keywords):

    #If no keywords provided, return the entire DataFrame
    if not keywords:
        return df

    #Combine keywords into one regex pattern, separated by the OR symbol '|'
    pattern = "|".join(keywords)

    #Use str.contains with case=False (ignore case) and na=False (exclude NaN)
    filtered_data = df[df["Answer"].str.contains(pattern, case=False, na=False)]

    return filtered_data

def free_text_search(filtered_data, query):
    final_results = filtered_data
    return final_results

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

    st.title("CogStack Search Mockup")

    # Create two columns: left is wider, right is narrower
    col_left, col_right = st.columns([3, 2], gap="large")

    # ---------------------
    # LEFT COLUMN: FREE TEXT
    # ---------------------
    with col_left:
        st.subheader("Free Text Search")
        free_text_query = st.text_area(
            label="",
            placeholder="Type your full text query here...",
            height=150
        )

        if st.button("Search"):
            #If user hasn't entered anything at all
            if not free_text_query.strip() and not st.session_state["keywords"]:
                st.warning("Please enter at least some text or a keyword.")
            else:
                #1) Filter by keywords
                filtered_data = keyword_search(st.session_state["keywords"])
                
                #2)Further refine by free text
                final_results = free_text_search(filtered_data, free_text_query.strip())
                st.subheader("Search Results")
                st.write(final_results)

    # ----------------------
    # RIGHT COLUMN: KEYWORDS
    # ----------------------
    with col_right:
        st.subheader("Keyword Search")
        #A small horizontal layout for the input + Add button
        kw_input_col, kw_add_btn_col = st.columns([3, 1], gap="small")

        with kw_input_col:
            st.text_input(
                label="Enter a single keyword",
                placeholder="E.g. 'python'",
                key="keyword_input",
                label_visibility="collapsed"
            )

        with kw_add_btn_col:
            #The button uses the callback to add/clear the keyword
            st.button("Add", on_click=add_keyword)

        st.write("Current Keywords:")
        st.write(st.session_state["keywords"])

if __name__ == "__main__":
    main()