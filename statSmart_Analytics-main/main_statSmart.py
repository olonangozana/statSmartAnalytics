import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib
import psycopg2
import numpy as np
matplotlib.use('Agg')
import sweetviz as sv
import dtale
import sqlalchemy
import load
import pandas_AI
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI
import openai
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_regression
from folium.plugins import FastMarkerCluster
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error,classification_report
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn import linear_model
from sklearn.feature_selection import VarianceThreshold
from sklearn import preprocessing
from sklearn.svm import SVC
import scipy.stats as stats
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import plotly.express as px
import plotly.io as pio
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Lasso, Ridge, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# st.set_page_config(initial_sidebar_state="collapsed")
import pandas as pd
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI
import joblib 
import pickle

import openai
import pandas as pd

image = Image.open('picture.png')

# st.image(image, caption='StatSmart App')

import hashlib
def make_hashes(passowrd):
    return hashlib.sha256(str.encode(passowrd)).hexdigest()

# Function to check if th passowrd hashed
def check_hashes(passowrd,hash_text):
    if make_hashes(passowrd)==hash_text:
        return hash_text
    return False

#Databasde Management
#Conneting to db
def init_connection():
    db_params = {
        'host': 'localhost',
        'dbname': 'postgres',
        'user': 'postgres',
        'password': '',
        'port':5432
    }
    
    try:
        connection = psycopg2.connect(**db_params)
        return connection
    except psycopg2.Error as e:
        print("Error connecting to the database:", e)
        return None
conn = init_connection()
cur = conn.cursor()

@st.cache_data(ttl=600)
def create_usertable():
    cur.execute('CREATE TABLE IF NOT EXISTS user_all(username TEXT,password TEXT);')
    
def add_userdata(username,password):
    hashed_password = password
    cur.execute('INSERT INTO user_all(username,password) VALUES(%s,%s)',(username,hashed_password))
    conn.commit()
    
# Login function
def login_user(username,password):
    cur.execute('SELECT * FROM user_all WHERE username=%s AND password=%s',(username,password))
    data=cur.fetchall()
    return data
    
def view_all_users():
    cur.execute('SELECT * FROM usertable')
    data=cur.fetchall()
    return data

### Main app functions
def main():

    st.title("StatSmart App")
    menu=["Home","About","Login","Signup"]
    password ="12345"
    username="Admin"
    choice=st.sidebar.selectbox("Menu",menu)

    if choice=='Home':
        st.subheader("Home")
      
    elif choice == "About":
        st.markdown("""**About the StatSmart Automated Data Analysis Application**
The StatSmart Automated Data Analysis Application is designed with the mission of revolutionizing the data analysis workflow for data scientists. Our software is meticulously crafted to streamline and simplify the entire process of data extraction, cleaning, and analysis, all while eliminating the need for manual intervention.

**Our Vision:**
Our vision is to empower data scientists and analysts with a powerful tool that automates the most time-consuming and repetitive tasks in the data analysis journey. By doing so, we aim to liberate data professionals from the drudgery of manual data handling and enable them to focus on the insights and decisions that truly matter.

**Key Features:**
- **Effortless Data Extraction:** Seamlessly import your datasets from various sources, saving you the hassle of manual data retrieval.

- **Smart Data Cleaning:** Our intelligent algorithms automatically identify and address data inconsistencies, missing values, and outliers, ensuring your data is pristine.

- **Advanced Analytics:** Dive into your data with a wide array of analytical tools and visualizations, gaining rapid insights and making data-driven decisions effortlessly.

- **Time Savings:** Say goodbye to hours spent on routine data tasks. With StatSmart, you'll achieve results in a fraction of the time.

**Why StatSmart?**
- **Efficiency:** Automate repetitive tasks and reduce analysis time, allowing you to focus on what truly matters.

- **Accuracy:** Our rigorous data cleaning processes enhance data quality, resulting in more reliable analyses.

- **User-Friendly:** Intuitive design and user-friendly interfaces make advanced data analysis accessible to all skill levels.

- **Flexibility:** Whether you're a beginner or an experienced data scientist, our application caters to your needs and adapts to your skill level.

**Join Us in Simplifying Data Analysis!**
StatSmart is your partner in unleashing the full potential of your data. Join our community of data scientists and analysts who are already experiencing the future of automated data analysis. Welcome to a new era of data-driven decision-making.

*StatSmart - DataScience in a Nutshell.*""")
      
    elif choice=="Login":
        st.subheader("Login Section")
        username=st.sidebar.text_input("User Name")
        password=st.sidebar.text_input("Password",type='password')

        if st.sidebar.checkbox("Login"):
            create_usertable()
            result=login_user(username,password)

            if result :
                    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
                    if uploaded_file is not None:
                        no_sidebar_style = """     
                        <style>         
                        div[data-testid="stSidebarNav"] {display: none;}     
                        </style> """ 
                        st.markdown(no_sidebar_style, unsafe_allow_html=True)
                    task=st.selectbox("Task",['Data Overview','Data Cleaning','Data Analysis','Machine Learning'])
          
                    if task == "Data Overview":
                        st.subheader("Overview")
                        df = load.load_csv(uploaded_file)
                        # my_report = dtale.show()
                        st.header('**Input Data**')
                        # st.write(df)

                        st.write('---')
                        st.header('**Data Overview**')

                        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["Dimensions", "Columns", "Unique Values","Missing Values", "Preview Dataset",
                                                    "Duplicated Rows", "Data Types", "Dataset Summary" ])

                        with tab1:
                            st.header("Show Dimensions")
                            st.write(df.shape)

                        with tab2:
                            st.header("Show Column Names")
                            st.write(df.columns)

                        with tab3:
                            st.header("Show Unique Values")
                            st.write(df.nunique())

                        with tab4:
                            st.header("Show Missing Values")
                            st.write(df.isna().sum())

                        with tab5:
                            st.header("Preview Dataset")
                            if st.button("Head"):
                                st.write(df.head())
                            elif st.button("Tail"):
                                st.write(df.tail())
                            else:
                                number =  st.slider("Select No of Rows",1, df.shape[0])
                                st.write(df.head(number))

                        with tab6:
                            st.header("Show Duplicated Rows")
                            duplicate_rows = df[df.duplicated()]
                            st.write(f"Number of Duplicate Rows: {len(duplicate_rows)}")

                        with tab7:
                            st.header("Data Types")
                            duplicate_rows = df[df.duplicated()]
                            data_types = df.dtypes
                            st.write(data_types)

                        with tab8:
                            st.header("Dataset Summary")
                            duplicate_rows = df[df.duplicated()]
                            st.write(df.describe())
                
                    
                        llm = OpenAI(api_token='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')

                        #initializing an instance of Pandas AI with openAI environment
                        pandas_ai = PandasAI(llm, verbose=True, conversational=False)

                        PROMPT = st.chat_input("Want more insight? Talk to me", key="insight_input_1")
                        
                        # using pandasAI instance to process text prompt and dataset
                        response = pandas_ai(df, PROMPT)
                        # printing the response
                        st.write(response)
              
                    elif task =='Data Cleaning':
                        if uploaded_file is not None:
                            df = load.load_csv(uploaded_file)
                            my_report = dtale.show()
                    
                        st.header('**Input DataFrame**')
                        st.write(df)
                        st.write('---')
                    
                            # Function to check for duplicates and give summary
                        def check_duplicates(df):
                            num_duplicates = df.duplicated().sum()
                            duplicate_percentage = (num_duplicates / len(df)) * 100
                            return num_duplicates, duplicate_percentage

                        # Function to handle missing data
                        def handle_missing_data(df):
                            missing_summary = df.isnull().sum()
                            missing_percentage = (missing_summary / len(df)) * 100
                            return missing_summary, missing_percentage

                        # Function to clean data based on user choices
                        def clean_data(df, column, method):
                            if method == "Remove Nulls":
                                df = df.dropna(subset=[column])
                            elif method == "Mean":
                                df[column].fillna(df[column].mean(), inplace=True)
                            elif method == "Median":
                                df[column].fillna(df[column].median(), inplace=True)
                            elif method == "Mode":
                                df[column].fillna(df[column].mode().iloc[0], inplace=True)
                                return df

                            # Streamlit App
                            st.title("Data Cleaning")
                            # Check for duplicates
                            st.header("1. Check for Duplicates")
                            num_duplicates, duplicate_percentage = check_duplicates(df)
                            st.write(f"Total Duplicates: {num_duplicates} ({duplicate_percentage:.2f}%)")
                                
                            # Option to delete duplicates
                            delete_duplicates = st.checkbox("Delete Duplicates?")
                            if delete_duplicates:
                                df.drop_duplicates(inplace=True)
                                st.write("Duplicates have been deleted from the dataset.")

                            # Handle missing data
                            st.header("2. Handling Missing Data")
                            missing_summary, missing_percentage = handle_missing_data(df)
                            for col in df.columns:
                                st.write(f"**{col}**: {missing_summary[col]} missing values ({missing_percentage[col]:.2f}%)")
                                if missing_percentage[col] < 5:
                                    clean_method = st.selectbox(f"Select cleaning method for {col}", ["Remove Nulls", "Mean", "Median", "Mode"])
                                    if clean_method != "Remove Nulls":
                                        df = clean_data(df, col, clean_method)
                                    
                            # Dealing with datatypes
                            st.header("3. Dealing with Datatypes")
                            change_datatype = st.checkbox("Change Datatypes?")
                            if change_datatype:
                                for col in df.columns:
                                    dtype_change = st.selectbox(f"Change datatype for {col}", ["No Change", "int64", "float64", "object"])
                                    if dtype_change != "No Change":
                                        try:
                                            df[col] = df[col].astype(dtype_change)
                                        except ValueError:
                                            st.warning(f"Unable to change datatype for {col}. Check if the data is compatible with the selected datatype.")
                                
                            #   Time Stamp
                            st.header("4. Change Timestamp")
                            extract_date = st.checkbox("Extract Date")
                            date_column = st.selectbox("Select a column with date:", df.columns)

                            if extract_date and date_column in df.columns:
                                df[date_column] = pd.to_datetime(df[date_column])
                                df['day_of_week'] = df[date_column].dt.day_name()
                                df['month'] = df[date_column].dt.month_name()
                                df['year'] = df[date_column].dt.year.astype(str).apply(lambda x: f"{x:0>4}")

                                def get_season(month):
                                    if month in ['December', 'January', 'February']:
                                        return 'Winter'
                                    elif month in ['March', 'April', 'May']:
                                        return 'Spring'
                                    elif month in ['June', 'July', 'August']:
                                        return 'Summer'
                                    else:
                                        return 'Autumn'

                            st.checkbox("Encode Day Of The Week")
                            selected_day = st.selectbox("Select a day of the week column to encode:", df.columns)
                            day_of_dict = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4,'Friday': 5, 'Saturday': 6, 'Sunday': 7}
                            if selected_day in df.columns:
                                df['day_numeric'] = df[selected_day].map(day_of_dict)
                                st.write(df[[selected_day, 'day_numeric']])  # Use 'selected_day' here
                            else:
                                st.warning(f"Column '{selected_day}' does not exist in the DataFrame.")
                                
                            st.checkbox("Encode Months")
                            selected_month = st.selectbox("Select a month column to encode:", df.columns)
                            month_dict = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6, 'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}
                            df['month_numeric'] = df[selected_month].map(month_dict)
                            if selected_month in df.columns:
                                df['month_numeric'] = df[selected_month].map(month_dict)
                                st.write(df[[selected_month, 'month_numeric']])
                            else:
                                st.warning(f"Column '{selected_month}' does not exist in the DataFrame.")
                                                
                            st.checkbox("Encode Seasons")
                            selected_season = st.selectbox("Select a season to encode:", df.columns)
                            season_dict = {'Winter': 1, 'Spring': 2, 'Summer': 3, 'Fall': 4}
                            df['season_numeric'] = df[selected_season].map(season_dict)
                            if selected_season in df.columns:
                                df['season_numeric'] = df[selected_season].map(season_dict)
                                st.write(df[[selected_season, 'season_numeric']])
                            else:
                                st.warning(f"Column '{selected_season}' does not exist in the DataFrame.")
                                
                            st.header("5. Drop Columns")
                            columns_to_drop = st.multiselect("Select columns to drop", df.columns)
                            column_drop = st.button("Drop Column")
                            if column_drop and columns_to_drop:
                                df.drop(columns=columns_to_drop, inplace=True)
                                st.write("Columns dropped:", columns_to_drop)
                            st.write("Updated DataFrame:", df)
                                
                            st.header("6. Rename Column")
                            def rename_column(df, old_col_name, new_col_name):
                                if old_col_name in df.columns:
                                    df.rename(columns={old_col_name: new_col_name}, inplace=True)
                            old_col_name = st.selectbox("Select the column to rename", df.columns)
                            new_col_name = st.text_input("Enter the new column name")
                            rename_button = st.button("Rename Column")
                            if rename_button and old_col_name and new_col_name:
                                rename_column(df, old_col_name, new_col_name)
                                st.write(f"Column '{old_col_name}' renamed to '{new_col_name}'")
                            # Display the updated DataFrame
                            st.write("Updated DataFrame:", df)

                                #Saving the Dataset
                            st.header("7. Save Dataset")  
                            csv_file_name = st.text_input("Enter CSV file name (without .csv extension):")
                            # Button to save to CSV
                            if st.button("Save The Dataset") and csv_file_name:
                                try:
                                    # Add .csv extension to the file name
                                    csv_file_name = f"{csv_file_name}.csv"
                                    # Save DataFrame to CSV
                                    df.to_csv(csv_file_name, index=False)
                                    st.success(f"Data has been successfully saved to {csv_file_name}")
                                except Exception as e:
                                    st.error(f"An error occurred: {str(e)}")      
                            llm = OpenAI(api_token='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')
                                #initializing an instance of Pandas AI with openAI environment
                            pandas_ai = PandasAI(llm, verbose=True, conversational=False)
                            PROMPT = st.chat_input("Want more insight? Talk to me", key="insight_input_2")    
                            # using pandasAI instance to process text prompt and dataset
                            response = pandas_ai(df, PROMPT)
                            # printing the response
                            st.write(response)
              
           
                    elif task== "Data Analysis":
                        filename = st.file_uploader("Upload Cleaned Dataset", type=("csv", "xlsx"))

                        # Pandas Profiling Report
                        if filename is not None:
                            df = pd.read_csv(filename)  
                            my_report = dtale.show(df)
                            # st.header('**Input DataFrame**')
                            # st.write(df)
                            st.write('---')
                    
                            st.header('**Data Visualization**')
                            st.subheader("Data Analytics")
                            analytics_option = st.selectbox("Select Analytics Type", ['Categorical', 'Numerical',])
                            st.set_option('deprecation.showPyplotGlobalUse', False)


                            fig = []
                            def explore_categorical(df):
                                categorical_cols = df.select_dtypes(include='object').columns.tolist()
                                selected_cols = st.multiselect("Select Categorical Columns for Comparison", categorical_cols, default=categorical_cols[:2], key="categorical_multiselect")

                                for col in selected_cols:
                                    plt.figure(figsize=(12, 6))
                                    plt.subplot(2, 2, 1)
                                    sns.countplot(data=df, x=col, order=df[col].value_counts().index, palette='Set2')
                                    plt.title(f'Distribution of {col}')
                                    plt.xlabel(col)
                                    plt.xticks(rotation=45)
                                    
                                    plt.subplot(2, 2, 2)
                                    proportions = df[col].value_counts(normalize=True)
                                    plt.pie(proportions, labels=proportions.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('Set2'))
                                    plt.title(f'Proportions of {col}')

                                    plt.tight_layout()
                                    st.pyplot(fig)
                          
                                return fig
                      
                            def explore_numerical(df):
                                numerical_cols = df.select_dtypes(include=['int', 'float']).columns.tolist()
                                # Select one or two columns for comparison
                                selected_cols = st.multiselect("Select Numerical Columns for Comparison", numerical_cols, default=numerical_cols[:2],key="numerical_multiselect")
                                for col in selected_cols:
                                    plt.figure(figsize=(12, 6))
                                    plt.subplot(2, 2, 1)
                                    sns.histplot(data=df, x=col, bins=20, kde=True, color='skyblue')
                                    plt.title(f'Distribution of {col}')
                                    plt.xlabel(col)

                                        # Boxplot
                                    plt.subplot(2, 2, 2)
                                    sns.boxplot(data=df, y=col, color='skyblue')
                                    plt.title(f'Boxplot of {col}')
                                    plt.ylabel(col)

                                    plt.tight_layout()
                                    st.pyplot(fig)
                                    
                                return fig
                    
                            if analytics_option == 'Categorical':
                                st.subheader("Categorical Analysis")
                                fig = explore_categorical(df)
                                st.pyplot(fig)
                            elif analytics_option == 'Numerical':
                                st.subheader("Numerical Analysis")
                                fig = explore_numerical(df)
                                st.pyplot(fig) 
                            llm = OpenAI(api_token='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')
                            #initializing an instance of Pandas AI with openAI environment
                            pandas_ai = PandasAI(llm, verbose=True, conversational=False)
                            PROMPT = st.chat_input("Want more insight? Talk to me", key="insight_input_3") 
                            # using pandasAI instance to process text prompt and dataset
                            response = pandas_ai(df, PROMPT)
                            # printing the response
                            st.write(response)
                            
                    else:
                            # Disable the deprecation warning
                            st.set_option('deprecation.showPyplotGlobalUse', False)
                            st.title("Machine Learning")
                            st.subheader("Upload a cleaned dataset")
                            #Upload the input file:
                            filename = st.file_uploader("Upload Cleaned Dataset", type=("csv", "xlsx"))
                            if filename is not None:
                                df = pd.read_csv(filename)  
                                my_report = dtale.show(df)
                            
                                
                                tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["Dimensions", "Columns", "Unique Values","Missing Values", "Preview Dataset",
                                                            "Duplicated Rows", "Data Types", "Dataset Summary" ])

                                with tab1:
                                    st.header("Show Dimensions")
                                    st.write(df.shape)

                                with tab2:
                                    st.header("Show Column Names")
                                    st.write(df.columns)

                                with tab3:
                                    st.header("Show Unique Values")
                                    st.write(df.nunique())

                                with tab4:
                                    st.header("Show Missing Values")
                                    st.write(df.isna().sum())

                                with tab5:
                                    st.header("Preview Dataset")
                                    if st.button("Head"):
                                        st.write(df.head())
                                    elif st.button("Tail"):
                                        st.write(df.tail())
                                    else:
                                        number =  st.slider("Select No of Rows",1, df.shape[0])
                                        st.write(df.head(number))

                                with tab6:
                                    st.header("Show Duplicated Rows")
                                    duplicate_rows = df[df.duplicated()]
                                    st.write(f"Number of Duplicate Rows: {len(duplicate_rows)}")

                                with tab7:
                                    st.header("Data Types")
                                    duplicate_rows = df[df.duplicated()]
                                    data_types = df.dtypes
                                    st.write(data_types)

                                with tab8:
                                    st.header("Dataset Summary")
                                    duplicate_rows = df[df.duplicated()]
                                    st.write(df.describe())
                                
                                with tab8:
                                    st.header("Show Infinites")
                                    st.write(df.isin([np.inf, -np.inf]).sum())

                                
                                st.subheader("FEATURE SELECTION")   
                                st.write("Feature selection is primarily focused on removing non-informative or redundant predictors from the model.")
                            
                                    # Calculate VIF for numerical features
                                numerical_features = df.select_dtypes(include=[np.number])
                                vif_data = pd.DataFrame()
                                vif_data["Feature"] = numerical_features.columns
                                vif_data["VIF"] = [variance_inflation_factor(numerical_features.values, i)
                                for i in range(numerical_features.shape[1])]
                            
                                    #   Plot VIF values
                                st.subheader("VIF (Variance Inflation Factor)")
                                st.write("The Variance Inflation Factor (VIF) measures how much the variance of an estimated regression coefficient increases when your predictors are correlated; it helps identify multicollinearity in regression analysis.")
                                fig, ax = plt.subplots(figsize=(8, 6))
                                sns.barplot(data=vif_data, x="VIF", y="Feature", ax=ax, orient="h")
                                st.pyplot(fig)
                            
                                st.subheader("Variance Threshold")
                                st.write("Variance threshold is a technique used to filter out low-variance features from a dataset, focusing on those with significant variance, which may contain more useful information for modeling.")
                            
                            
                                #  Filter and select numerical columns
                                numerical_columns = df.select_dtypes(include=['number']).columns

                                # Add a slider for variance threshold
                                variance_threshold = st.slider("Select Variance Threshold", 0.0, 2.0, 1.0, 0.1)

                                # Allow the user to pick columns for variance threshold
                                selected_columns = st.multiselect("Select Numerical Columns for Variance Threshold", numerical_columns)

                                if len(selected_columns) == 0:
                                    st.warning("Please select at least one numerical column for variance threshold.")
                                else:
                                    X = df[selected_columns]
                                    v_threshold = VarianceThreshold(threshold=variance_threshold)  # Use the selected threshold
                                    v_threshold.fit_transform(X)

                                    # Update selected columns based on the threshold
                                    selected_features = v_threshold.get_support()
                            
                                    # Create a DataFrame with column names and their True/False values
                                    selected_df = pd.DataFrame({'Column Name': selected_columns, 'Meets Threshold': selected_features})

                                    # Display the DataFrame
                                    st.write("Selected Features based on Variance Threshold:")
                                    st.write(selected_df)
                                
                            # Title and subheader
                                st.title("Chi-Squared Test for Categorical Variables")
                                st.subheader("Assessing Association Between Categorical Columns")
                                st.write("The Chi-Square test is a statistical procedure for determining the difference between observed and expected data. This test can also be used to determine whether it correlates to the categorical variables in our data.")

                                # Display dataset preview
                                st.subheader("Dataset Preview")
                                st.write(df.head())

                                # Allow the user to select two categorical columns
                                categorical_columns = df.select_dtypes(include=['object']).columns
                                col1 = st.selectbox("Select the first categorical column", categorical_columns)
                                col2 = st.selectbox("Select the second categorical column", categorical_columns)

                                # Create a contingency table
                                contingency_table = pd.crosstab(df[col1], df[col2])

                                # Perform the chi-squared test
                                chi2, p, _, _ = stats.chi2_contingency(contingency_table)

                                # Display results
                                st.subheader("Chi-Squared Test Results")
                                st.write(f"Chi-squared statistic: {chi2}")
                                st.write(f"P-value: {p}")

                                # Interpret the results
                                alpha = 0.05  # Significance level
                                if p <= alpha:
                                    st.write("There is a significant association between the selected categorical columns.")
                                else:
                                    st.write("There is no significant association between the selected categorical columns.")
                                
                                st.subheader("SCATTER PLOT")
                                st.write("It is used to visualize the relationship or correlation between these variables.")
                            
                                #visualization
                                #scatter plot
                                col1 = st.selectbox('Which feature on x?', df.columns)
                                col2 = st.selectbox('Which feature on Y?', df.columns)
                                fig = px.scatter(df, x = col1, y = col2)
                                st.plotly_chart(fig)

                                #correlation plot
                                st.subheader("CORRELATION PLOT")
                                st.write("A correlation plot visualizes the strength and direction of relationships between variables, with values near 1 for strong positive correlations, -1 for strong negative correlations, and near 0 for no correlation.")
                                if st.checkbox("Show Correlation plots with Seaborn"):
                                #  sns_plot = sns.heatmap(data.corr())
                                #  st.pyplot(fig)
                                    st.write(sns.heatmap(df.corr(numeric_only=True)))
                                    st.pyplot()

                                # Add an option to save the plot as an image
                                if st.button("Save Plot as Image"):
                                    filename = "scatter_plot.png"
                                    pio.write_image(fig, filename)

                                # Provide a download link for the saved image
                                st.markdown(f"Download the saved plot as [**{filename}**](./{filename})")
                                
                                # Machine learning Algorithm
                                st.subheader("MACHINE LEARNING MODELING")
                                st.write("Machine learning modeling involves using algorithms to build predictive models for classification (categorizing data into classes) and regression (predicting numerical values) tasks based on input features and training data.")
                                problem_type = st.radio("Select Problem Type", ["Classification", "Regression"])
                                if problem_type == "Classification":
                                    alg = [
                                        "XGBoost Classifier",
                                        "Support Vector Machine",
                                        "Random Forest Classifier",
                                        "Logistic Regression",
                                        "Adaboost Classifier",
                                        "Decision Tree Classifier"
                                        ]
                                else:
                                    alg = [
                                        "Lasso",
                                        "Random Forest Regressor",
                                        "Linear Regression",
                                        "Decision Tree Regressor",
                                        "Ridge",
                                        "Stacking Regressor"
                                        ]
                                model_name = st.selectbox(f"Select {'Classification' if problem_type == 'Classification' else 'Regression'} Algorithm", alg)

                                if problem_type == "Classification":
                                    models = {
                                        "XGBoost Classifier": XGBClassifier(),
                                        "Support Vector Machine": SVC(),
                                        "Random Forest Classifier": RandomForestClassifier(),
                                        "Logistic Regression": LogisticRegression(),
                                        "Adaboost Classifier": AdaBoostClassifier(),
                                        "Decision Tree Classifier": DecisionTreeClassifier()}
                                else:
                                    models = {
                                        "Lasso": Lasso(alpha=1.0),
                                        "Random Forest Regressor": RandomForestRegressor(),
                                        "Linear Regression": LinearRegression(),
                                        "Decision Tree Regressor": DecisionTreeRegressor(),
                                        "Ridge": Ridge(alpha=1.0),
                                        "Stacking Regressor": StackingRegressor(
                                        estimators=[("model1", RandomForestRegressor()),
                                                    ("model2", DecisionTreeRegressor()),],
                                        final_estimator=LinearRegression(),),
                                        }
                
                                # Label encode categorical columns
                                label_encoders = {}
                                categorical_columns = df.select_dtypes(include=[object]).columns

                                # categorical_columns = data.select_dtypes(include=[np.object]).columns

                                for col in categorical_columns:
                                    le = LabelEncoder()
                                    df[col] = le.fit_transform(df[col])
                                    label_encoders[col] = le 
                    
                                features = st.multiselect("Select Feature Columns",df.columns) 
                                labels = st.multiselect("Select Target Columns",df.columns)         

                                features = df[features].values
                                labels  = df[labels].values

                                test_percent = st.slider("Select % to test the model", 1 , 100)
                                test_percent = test_percent/100

                                X_train,X_test,y_train,y_test = train_test_split(features,labels,train_size=test_percent,random_state=1)    

                                scaling_option = st.radio("Select Scaling Option", ("None", "StandardScaler", "MinMaxScaler"))

                                if scaling_option == "StandardScaler":
                                    scaler = StandardScaler()
                                    X_train_scaled = scaler.fit_transform(X_train)
                                    X_test_scaled = scaler.transform(X_test)
                                elif scaling_option == "MinMaxScaler":
                                    scaler = MinMaxScaler()
                                    X_train_scaled = scaler.fit_transform(X_train)
                                    X_test_scaled = scaler.transform(X_test)
                                else:
                                # No scaling
                                    X_train_scaled = X_train
                                    X_test_scaled = X_test
                                    
                                # Train and Evaluate the Model
                                if st.button("Train and Evaluate"):
                                    if model_name in models:
                                        model = models[model_name]
                                        model.fit(X_train, y_train)
                                        if problem_type == "Classification":
                                            y_pred = model.predict(X_test)
                                            report = classification_report(y_test, y_pred) 
                                        else:
                                            y_pred = model.predict(X_test)
                                            mse = mean_squared_error(y_test, y_pred)
                                            rmse = np.sqrt(mse) 
                                            mae = mean_absolute_error(y_test, y_pred)
                                            r2 = r2_score(y_test, y_pred)
                                            report = (
                                                f"Mean Squared Error (MSE): {mse:.2f}\n"
                                                f"Root Mean Squared Error (RMSE): {rmse:.2f}\n"
                                                f"Mean Absolute Error (MAE): {mae:.2f}\n"
                                                f"R-squared (R2): {r2:.2f}"
                                                    )

                                        st.write(f"**{model_name} Model Evaluation**")
                                        st.text(report)
                                        model_name = st.text_input("Enter the model name (without extension):")
                                        if st.button("Save Model") and model_name:
                                            filename = f"{model_name}_model.pkl"
                                            with open(filename, 'wb') as file:
                                                pickle.dump(model, file)
                                            st.success(f"{model_name} model saved as {filename}")
                                    else:
                                        st.write("Select a valid algorithm.")
                                    
                            llm = OpenAI(api_token='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')
                            #initializing an instance of Pandas AI with openAI environment
                            pandas_ai = PandasAI(llm, verbose=True, conversational=False)
                            PROMPT = st.chat_input("Want more insight? Talk to me", key="insight_input_4")
                            # using pandasAI instance to process text prompt and dataset
                            response = pandas_ai(df, PROMPT)
                            # printing the response
                            st.write(response)                   
                         
            else:
                return st.warning("Incorrect Username/Password")                     
                    
    elif choice=="Signup":
            st.subheader("Create New Account")
            new_user=st.text_input("Username")
            new_password=st.text_input("Password",type='password')

            if st.button("Signup"):
                create_usertable()
                add_userdata(new_user,new_password)
                st.success("You have succesfully created a valid Acccount")
                st.info("Go to Login Menu to login")

if __name__ == '__main__':
    main()

            


