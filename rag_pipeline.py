import faiss
import pickle
import os
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv


INDEX_PATH = "data/product_index.faiss"
DOC_PATH = "data/documents.pkl"


# -----------------------------
# 1 Load embedding model
# -----------------------------
def load_embedding_model():
    model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    return model


# -----------------------------
# 2 Load FAISS index
# -----------------------------
def load_index(path):
    return faiss.read_index(path)


# -----------------------------
# 3 Load documents
# -----------------------------
def load_documents(path):
    with open(path, "rb") as f:
        documents = pickle.load(f)
    return documents


# -----------------------------
# 4 Encode query
# -----------------------------
def encode_query(model, query):
    return model.encode([query])


# -----------------------------
# 5 Retrieve context
# -----------------------------
def retrieve_context(index, documents, query_embedding, k=3):

    distances, indices = index.search(query_embedding, k)

    context_docs = []

    for idx in indices[0]:
        context_docs.append(documents[idx])

    return context_docs


# -----------------------------
# 6 Build prompt
# -----------------------------
def build_prompt(query, contexts):

    context_text = "\n\n".join(contexts)

    prompt = f"""
You are a helpful shopping assistant.

Use ONLY the information from the product catalog below.

Context:
{context_text}

User question:
{query}

Instructions:
- Recommend suitable products.
- Use a numbered list format: 1., 2., 3.
- Do NOT use bullet points like * or -.
- Each recommendation should include product name, material, and price.
- Give a short explanation.

Example format:

1. Product Name - Material - Price
   Short explanation.

2. Product Name - Material - Price
   Short explanation.
"""

    return prompt


# -----------------------------
# 7 Generate answer with Gemini
# -----------------------------
def generate_answer(prompt):
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")

    genai.configure(api_key=api_key)

    model = genai.GenerativeModel("gemini-2.5-flash")

    response = model.generate_content(prompt)

    return response.text


# -----------------------------
# Main pipeline
# -----------------------------
def main():

    print("Loading embedding model...")
    embedding_model = load_embedding_model()

    print("Loading FAISS index...")
    index = load_index(INDEX_PATH)

    print("Loading documents...")
    documents = load_documents(DOC_PATH)

    query = input("Enter your question: ")

    query_embedding = encode_query(embedding_model, query)

    contexts = retrieve_context(index, documents, query_embedding, k=3)

    prompt = build_prompt(query, contexts)

    print("\nGenerating answer...\n")

    answer = generate_answer(prompt)

    print("Answer:")
    print(answer)


if __name__ == "__main__":
    main()