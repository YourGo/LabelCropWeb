import streamlit as st

st.write("Python script is executing!")
st.write("Python version:", __import__("sys").version)
st.write("Working directory:", __import__("os").getcwd())
st.write("Script completed successfully!")
