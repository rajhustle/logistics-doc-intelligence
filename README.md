
# UltraShip RAG Project

## Install

pip install -r requirements.txt

## Run

streamlit run app.py

## Features

- Document upload
- Question answering
- Guardrails
- Confidence scoring
- Structured extraction

## Architecture

Upload → Chunk → Embed → FAISS → Retrieve → Guardrail → Answer
