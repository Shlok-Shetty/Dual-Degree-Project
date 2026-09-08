"""Streamlit entry point for the lite (no-LLM) version.

Run:
    streamlit run comparison-search-app-lite/run_app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.streamlit_app import main

if __name__ == "__main__":
    main()