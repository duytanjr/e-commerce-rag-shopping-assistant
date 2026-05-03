import faiss
import pickle
from sentence_transformers import SentenceTransformer


INDEX_PATH = "data/product_index.faiss"
DOC_PATH = "data/documents.pkl"


# -----------------------------
# 1. Load embedding model
# -----------------------------
def load_embedding_model():
    model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    return model


# -----------------------------
# 2. Load FAISS index
# -----------------------------
def load_index(path):
    index = faiss.read_index(path)
    return index


# -----------------------------
# 3. Load documents
# -----------------------------
def load_documents(path):
    with open(path, "rb") as f:
        documents = pickle.load(f)
    return documents


# -----------------------------
# 4. Encode query
# -----------------------------
def encode_query(model, query):
    query_embedding = model.encode([query])
    return query_embedding


# -----------------------------
# 5. Retrieve top-k documents
# -----------------------------
def retrieve_top_k(index, documents, query_embedding, k=3):

    distances, indices = index.search(query_embedding, k)

    results = []

    for idx in indices[0]:
        results.append(documents[idx])

    return results


# -----------------------------
# Main pipeline
# -----------------------------
def main():

    print("Loading model...")
    model = load_embedding_model()

    print("Loading FAISS index...")
    index = load_index(INDEX_PATH)

    print("Loading documents...")
    documents = load_documents(DOC_PATH)

    query = input("Enter your query: ")

    print("Encoding query...")
    query_embedding = encode_query(model, query)

    print("Retrieving documents...")
    results = retrieve_top_k(index, documents, query_embedding, k=3)

    print("\nTop Results:\n")

    for i, doc in enumerate(results):
        print(f"Result {i+1}")
        print(doc)
        print("-" * 50)


if __name__ == "__main__":
    main()