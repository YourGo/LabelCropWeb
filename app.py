# Try to cause an immediate error to see if script executes
raise Exception("TEST: Script is executing - this should appear in logs")

import sys
import os

# Write to file in current directory for debugging
debug_file = os.path.join(os.getcwd(), 'debug.log')
try:
    with open(debug_file, 'w') as f:
        f.write(f"Python executable: {sys.executable}\n")
        f.write(f"Python version: {sys.version}\n")
        f.write(f"Working directory: {os.getcwd()}\n")
        f.write("Testing basic imports...\n")
except Exception as e:
    # If we can't write to file, try to raise an exception to see it in logs
    raise Exception(f"Failed to write debug file: {e}")

try:
    import streamlit as st
    with open(debug_file, 'a') as f:
        f.write("✓ streamlit imported successfully\n")
        f.write(f"Streamlit version: {st.__version__}\n")
except Exception as e:
    with open(debug_file, 'a') as f:
        f.write(f"✗ streamlit import failed: {e}\n")
    raise Exception(f"Streamlit import failed: {e}")

with open(debug_file, 'a') as f:
    f.write("App starting...\n")

st.set_page_config(page_title="Test App", page_icon="🧪", layout="wide")
st.title("Test App")
st.write("If you see this, the app started successfully!")

with open(debug_file, 'a') as f:
    f.write("App UI rendered\n")

