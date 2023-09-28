import streamlit as st
import pandas as pd

uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
                    @st.cache_data
                    def load_csv():
                        csv = pd.read_csv(uploaded_file)
                        return csv
                    df = load_csv()
