import pandas as pd
import numpy as np
import faiss
import pickle
import re
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


DATA_PATH = "data/rag_dataset.csv"
INDEX_PATH = "data/product_index.faiss"
DOC_PATH = "data/documents.pkl"
BM25_PATH = "data/bm25_model.pkl"


# -----------------------------
# 1. Load dataset
# -----------------------------
def load_dataset(path):
    df = pd.read_csv(path)
    return df


# -----------------------------
# 2. Create RAG text
# -----------------------------
def create_rag_documents(df):
    documents = []

    for _, row in df.iterrows():
        text = (
            f"Product Name: {row['product_name']}. "
            f"Brand: {row['brand']}. "
            f"Category: {row['category']}. "
            f"Material: {row['material']}. "
            f"Color: {row['color']}. "
            f"Price: ${row['price_usd']}. "
            f"Description: {row['description']}"
        )
        documents.append(text)

    return documents


# -----------------------------
# 3. Load embedding model
# -----------------------------
def load_embedding_model():
    model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    return model


# -----------------------------
# 4. Generate embeddings
# -----------------------------
def generate_embeddings(model, documents):
    embeddings = model.encode(documents, show_progress_bar=True)
    embeddings = np.array(embeddings)
    return embeddings


# -----------------------------
# 5. Build FAISS index
# -----------------------------
def build_faiss_index(embeddings):

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    return index


# -----------------------------
# 6. Build BM25 index
# -----------------------------
def tokenize_for_bm25(text):
    return re.findall(r"\w+", text.lower())


def build_bm25_index(documents):
    tokenized_corpus = [tokenize_for_bm25(doc) for doc in documents]
    return BM25Okapi(tokenized_corpus)


# -----------------------------
# 7. Save index + documents + bm25
# -----------------------------
def save_artifacts(index, documents, bm25_index):

    faiss.write_index(index, INDEX_PATH)

    with open(DOC_PATH, "wb") as f:
        pickle.dump(documents, f)

    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25_index, f)


# -----------------------------
# Main pipeline
# -----------------------------
def main():

    print("Loading dataset...")
    df = load_dataset(DATA_PATH)

    print("Creating RAG documents...")
    documents = create_rag_documents(df)

    print("Loading embedding model...")
    model = load_embedding_model()

    print("Generating embeddings...")
    embeddings = generate_embeddings(model, documents)

    print("Building FAISS index...")
    index = build_faiss_index(embeddings)

    print("Building BM25 index...")
    bm25_index = build_bm25_index(documents)

    print("Saving artifacts...")
    save_artifacts(index, documents, bm25_index)

    print("Index built successfully!")
    print("Total documents:", len(documents))


if __name__ == "__main__":
    main()