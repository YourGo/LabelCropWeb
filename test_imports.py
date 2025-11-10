#!/usr/bin/env python3
print("Testing imports...")

try:
    import streamlit as st
    print("✓ streamlit imported")
except Exception as e:
    print(f"✗ streamlit import failed: {e}")

try:
    import cv2
    print("✓ cv2 imported")
except Exception as e:
    print(f"✗ cv2 import failed: {e}")

try:
    import numpy as np
    print("✓ numpy imported")
except Exception as e:
    print(f"✗ numpy import failed: {e}")

try:
    from PIL import Image
    print("✓ PIL imported")
except Exception as e:
    print(f"✗ PIL import failed: {e}")

try:
    import fitz
    print("✓ fitz imported")
except Exception as e:
    print(f"✗ fitz import failed: {e}")

try:
    from pyzbar.pyzbar import decode as zbar_decode
    print("✓ pyzbar imported")
except Exception as e:
    print(f"✗ pyzbar import failed: {e}")

try:
    from label_processor import PDFLabelProcessor
    print("✓ label_processor imported")
except Exception as e:
    print(f"✗ label_processor import failed: {e}")

print("Import testing complete.")