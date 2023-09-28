import pandas as pd
import streamlit as st
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI
import chardet 
import openai
import load


llm = OpenAI(api_token='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')
pandas_ai = PandasAI(llm, verbose=True, conversational=False)

def process_uploaded_csv(uploaded_file):
    try:
        raw_data = uploaded_file.read()
        encoding = chardet.detect(raw_data)['encoding']

        df = pd.read_csv(uploaded_file, encoding=encoding)

        st.write(f"Data from {uploaded_file.name}:")
        st.dataframe(df)

    except Exception as e:
        st.error(f"Error reading {uploaded_file.name}: {e}")

st.header("Upload CSV Files")
uploaded_files = st.file_uploader("Upload your input CSV file(s)", type=["csv"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        process_uploaded_csv(uploaded_file)

st.text("Note: This app supports multiple CSV files with any UTF encoding.")

    


