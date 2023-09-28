import streamlit as st
import pandas as pd

@st.cache(allow_output_mutation=True)
def load_csv(uploaded_file):
    @st.cache
    def load_csv_data():
        csv = pd.read_csv(uploaded_file)
        return csv
        return data 
    return load_csv_data()

