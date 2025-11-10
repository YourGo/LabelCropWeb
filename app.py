import sys
import os

# Write to file for debugging
with open('/tmp/debug.log', 'w') as f:
    f.write(f"Python executable: {sys.executable}\n")
    f.write(f"Python version: {sys.version}\n")
    f.write(f"Working directory: {os.getcwd()}\n")
    f.write("Testing basic imports...\n")

try:
    import streamlit as st
    with open('/tmp/debug.log', 'a') as f:
        f.write("✓ streamlit imported successfully\n")
        f.write(f"Streamlit version: {st.__version__}\n")
except Exception as e:
    with open('/tmp/debug.log', 'a') as f:
        f.write(f"✗ streamlit import failed: {e}\n")
    sys.exit(1)

with open('/tmp/debug.log', 'a') as f:
    f.write("App starting...\n")

st.set_page_config(page_title="Test App", page_icon="🧪", layout="wide")
st.title("Test App")
st.write("If you see this, the app started successfully!")

with open('/tmp/debug.log', 'a') as f:
    f.write("App UI rendered\n")

