# BT

A collection of Streamlit apps for research paper summarization plus a small Neo4j Cypher generator script.

## Setup
1. Create a virtual environment.
2. Install dependencies:

   pip install -r requirements.txt

## Streamlit apps
Run any of these with Streamlit:

- streamlit run integration.py
- streamlit run app.py
- streamlit run summ.py
- streamlit run summ_ollama.py

## Notes
- Ollama must be running for the Llama apps. Make sure the model is available.
- Large artifacts and local data are gitignored (venv, node_modules, data, logs, figures, model folders, PDFs).
