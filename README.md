# E-commerce RAG Shopping Assistant

An end-to-end Retrieval-Augmented Generation (RAG) project for product recommendation in an e-commerce setting.
The current version uses a Streamlit web app and a hybrid retrieval strategy that combines vector search and keyword search before generating answers with Gemini.

## Key Features

- Streamlit web interface for interactive shopping Q&A.
- Hybrid Search retrieval as the default retrieval path.
- FAISS vector search for semantic similarity matching.
- BM25 keyword search for lexical relevance.
- Reciprocal Rank Fusion (RRF) to merge vector and BM25 rankings.
- Multilingual Embeddings: Powered by sentence-transformers (paraphrase-multilingual-MiniLM-L12-v2) to handle diverse queries accurately.
- Gemini API integration for final natural-language recommendations.
- Modular Architecture: Clean separation of concerns between data indexing (offline) and querying (online).


## Architecture Overview

The project is organized into two phases:

1. Offline indexing (`build_index.py`)
   - Load the catalog dataset from `data/rag_dataset.csv`.
   - Build product documents used by both retrievers.
   - Generate embeddings and build `data/product_index.faiss`.
   - Build BM25 model from the same document list and save `data/bm25_model.pkl`.
   - Save text documents to `data/documents.pkl`.

2. Online retrieval + generation (`app.py` + `rag_pipeline.py`)
   - Streamlit receives the user query.
   - Query is encoded with SentenceTransformers.
   - Hybrid retrieval runs:
     - FAISS vector search
     - BM25 keyword search
     - RRF fusion
   - Top contexts are inserted into a prompt.
   - Gemini generates the final recommendation response.

## Project Structure

```text
Naive_RAG/
├── app.py                     # Streamlit web app (main entrypoint)
├── build_index.py             # Offline indexing for FAISS + BM25 artifacts
├── rag_pipeline.py            # Core retrieval + prompt + generation pipeline
├── requirements.txt           # Python dependencies
├── .env                       # Environment variables (Gemini API key)
└── data/
    ├── rag_dataset.csv        # Source product dataset
    ├── product_index.faiss    # FAISS vector index artifact
    ├── documents.pkl          # Serialized document list artifact
    └── bm25_model.pkl         # Serialized BM25 model artifact
```

## Tech Stack

- Python
- Streamlit
- SentenceTransformers (`paraphrase-multilingual-MiniLM-L12-v2`)
- FAISS (vector index)
- rank-bm25 (keyword retrieval)
- Google Gemini API (`gemini-2.5-flash`)
- python-dotenv

## Usage

1. Clone the project

```bash
git clone <your-repo-url>
cd Naive_RAG
```

2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Create `.env` and add your Gemini API key

```env
GEMINI_API_KEY=your_api_key_here
```

5. Build or rebuild retrieval artifacts

```bash
python3 build_index.py
```

This command generates/updates:
- `data/product_index.faiss`
- `data/documents.pkl`
- `data/bm25_model.pkl`

6. Run the Streamlit app

```bash
streamlit run app.py
```

## Example Usage (Streamlit)

After starting Streamlit, open the local URL shown in your terminal (commonly `http://localhost:8501`) and try prompts such as:

- `Recommend a blue cotton shirt for summer`
- `What are some affordable jeans?`
- `Do you have waterproof hiking jackets?`

The app will retrieve product context with Hybrid Search (FAISS + BM25 + RRF) and then generate a recommendation using Gemini.

## Notes

- If retrieval artifacts are missing or outdated, rerun:

```bash
python3 build_index.py
```