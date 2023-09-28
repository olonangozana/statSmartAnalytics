#LOGIN & signup
# from ast import main
# import numpy as np
# import pandas as pd
# import streamlit as st
# #from pandas_profiling import ProfileReport
# from ydata_profiling import ProfileReport
# from streamlit_pandas_profiling import st_profile_report

# # Security
# #passlib,hashlib,bcrypt,scrypt
# import hashlib
# def make_hashes(password):
# 	return hashlib.sha256(str.encode(password)).hexdigest()

# def check_hashes(password,hashed_text):
# 	if make_hashes(password) == hashed_text:
# 		return hashed_text
# 	return False
# # DB Management
# import sqlite3 
# conn = sqlite3.connect('data.db')
# c = conn.cursor()
# # DB  Functions
# def create_usertable():
# 	c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')


# def add_userdata(username,password):
# 	c.execute('INSERT INTO userstable(username,password) VALUES (?,?)',(username,password))
# 	conn.commit()

# def login_user(username,password):
# 	c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',(username,password))
# 	data = c.fetchall()
# 	return data


# def view_all_users():
# 	c.execute('SELECT * FROM userstable')
# 	data = c.fetchall()
# 	return data



# def main():
# 	"""Simple Login App"""

# 	st.title("The statSmart Analytics App")

# 	menu = ["Login","SignUp"]
# 	choice = st.sidebar.selectbox("Menu",menu)

# 	if choice == "Login":
# 		st.subheader("Login Section")

# 		username = st.sidebar.text_input("User Name")
# 		password = st.sidebar.text_input("Password",type='password')
# 		if st.sidebar.checkbox("Login"):
# 			# if password == '12345':
# 			create_usertable()
# 			hashed_pswd = make_hashes(password)

# 			result = login_user(username,check_hashes(password,hashed_pswd))
# 			if result:

# 				st.success("Logged In as {}".format(username)) 

# 				task = st.selectbox("Task",["Add Post","Analytics","Profiles"])
# 				if task == "Add Post":
# 					st.subheader("Add Your Post")

# 				elif task == "Analytics":
# 					st.subheader("Analytics")
# 				elif task == "Profiles":
# 					st.subheader("User Profiles")
# 					user_result = view_all_users()
# 					clean_db = pd.DataFrame(user_result,columns=["Username","Password"])
# 					st.dataframe(clean_db)
# 			else:
# 				st.warning("Incorrect Username/Password")





# 	elif choice == "SignUp":
# 		st.subheader("Create New Account")
# 		new_user = st.text_input("Username")
# 		new_password = st.text_input("Password",type='password')

# 		if st.button("Signup"):
# 			create_usertable()
# 			add_userdata(new_user,make_hashes(new_password))
# 			st.success("You have successfully created a valid Account")
# 			st.info("Go to Login Menu to login")



# if __name__ == '__main__':
# 	main()












#CSV LOADING


from ast import main
import numpy as np
import pandas as pd
import streamlit as st
#from pandas_profiling import ProfileReport
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report



# Web App Title
st.markdown('''
**The statSmart Analytics APP**
''')

# Upload CSV data
with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])


# Pandas Profiling Report
if uploaded_file is not None:
    @st.cache_data
    def load_csv():
        csv = pd.read_csv(uploaded_file)
        return csv
    df = load_csv()
    pr = ProfileReport(df, explorative=True)
    st.header('**Input DataFrame**')
    st.write(df)
    st.write('---')
    st.header('**Pandas Profiling Report**')
    st_profile_report(pr)
else:
    st.info('Awaiting for CSV file to be uploaded.')
   










#EXAMPLE OF VISUALISATIONS


# import pandas as pd
# import streamlit as st
# from PIL import Image

# # Web App Title
# st.markdown('''
# **The statSmart Analytics APP**
# ''')

# # Function to load CSV data
# @st.cache(allow_output_mutation=True)
# def load_csv(file):
#     csv = pd.read_csv(file)
#     return csv
# st.title("DATA ANALYTICS")

# def display_four_images(image_paths):
#     # Display the images in two columns
#     cols = st.columns(2)
#     for i, image_path in enumerate(image_paths):
#         cols[i % 2].image(Image.open(image_path), caption="Original Image", use_column_width=True)
        

# def main():
#     st.title("DATA ANALYTICS")

# option = st.selectbox("Choose an option:", [ "Analysis 1", "Analysis 2", "Analysis 3"])

# if option == "Analysis 1":
#         st.header("Analysis 1 ")
#         image_paths_option1 = ["/Users/da_m1_39/Downloads/Alvin_Smart_Money_ClassificationML-Zama_ML/test1.png",
#        						   "/Users/da_m1_39/Downloads/Alvin_Smart_Money_ClassificationML-Zama_ML/test2.png",
#         					 "/Users/da_m1_39/Downloads/Alvin_Smart_Money_ClassificationML-Zama_ML/test3.png",
#        							 "/Users/da_m1_39/Downloads/Alvin_Smart_Money_ClassificationML-Zama_ML/test4.png",
# ]
#         display_four_images(image_paths_option1)
# elif option == "Analysis 2":
#         st.header("Analysis 2 ")
#         image_paths_option2 = ["/Users/da_m1_39/Downloads/Alvin_Smart_Money_ClassificationML-Zama_ML/test5.png",
#        						   "/Users/da_m1_39/Downloads/Alvin_Smart_Money_ClassificationML-Zama_ML/test6.png",
#         					 "/Users/da_m1_39/Downloads/Alvin_Smart_Money_ClassificationML-Zama_ML/test7.png",
#        							 "/Users/da_m1_39/Downloads/Alvin_Smart_Money_ClassificationML-Zama_ML/test8.png",
#     ]
#         display_four_images(image_paths_option2)

# elif option == "Analysis 3":
#         st.header("Analysis 3 ")
#         image_paths_option3 = ["/Users/da_m1_39/Downloads/Alvin_Smart_Money_ClassificationML-Zama_ML/test9.png",
#        						   "/Users/da_m1_39/Downloads/Alvin_Smart_Money_ClassificationML-Zama_ML/test10.png",
#         					 "/Users/da_m1_39/Downloads/Alvin_Smart_Money_ClassificationML-Zama_ML/test11.png",
#        							 "/Users/da_m1_39/Downloads/Alvin_Smart_Money_ClassificationML-Zama_ML/test12.png",
#     ]

#         display_four_images(image_paths_option3)
        
        

# if __name__ == "__main__":
#     main()

























