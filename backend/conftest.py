import os
import sys

# Ensure the backend directory is on sys.path so that 'app' package can be imported
BACKEND_DIR = os.path.abspath(os.path.dirname(__file__))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
