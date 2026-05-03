import pandas as pd
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer


DATA_PATH = "data/rag_dataset.csv"
INDEX_PATH = "data/product_index.faiss"
DOC_PATH = "data/documents.pkl"


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
# 6. Save index + documents
# -----------------------------
def save_artifacts(index, documents):

    faiss.write_index(index, INDEX_PATH)

    with open(DOC_PATH, "wb") as f:
        pickle.dump(documents, f)


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

    print("Saving artifacts...")
    save_artifacts(index, documents)

    print("Index built successfully!")
    print("Total documents:", len(documents))


if __name__ == "__main__":
    main()