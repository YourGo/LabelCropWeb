import sys
print("Python executable:", sys.executable)
print("Python version:", sys.version)
print("Working directory:", __import__('os').getcwd())

print("Testing basic imports...")
try:
    import streamlit as st
    print("✓ streamlit imported successfully")
except Exception as e:
    print(f"✗ streamlit import failed: {e}")
    sys.exit(1)

print("Streamlit version:", st.__version__)
print("App starting...")

st.set_page_config(page_title="Test App", page_icon="🧪", layout="wide")
st.title("Test App")
st.write("If you see this, the app started successfully!")

