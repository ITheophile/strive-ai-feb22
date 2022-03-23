import streamlit as st

st.title('My Web App')


data = st.sidebar.selectbox('Select Data', ['data1', 'data2'])

st.write(data)

print(data)
