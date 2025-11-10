print("Starting app.py")
print("Python version:", __import__('sys').version)
print("Working directory:", __import__('os').getcwd())

# Minimal Streamlit app
import streamlit as st
st.write("Hello from Streamlit Cloud!")

