"""Streamlit entry point.

Run from the folder containing this file:
    streamlit run comparison-search-app/run_app.py

Or with an absolute path from anywhere:
    streamlit run C:\\Users\\shlok\\projects\\ddp-llm\\comparison-search-app\\run_app.py
"""
import sys
from pathlib import Path

# make the `app` package importable regardless of where streamlit was launched
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.streamlit_app import main

if __name__ == "__main__":
    main()
