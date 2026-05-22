import faiss
import pickle
import os
import re
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv


INDEX_PATH = "data/product_index.faiss"
DOC_PATH = "data/documents.pkl"
BM25_PATH = "data/bm25_model.pkl"


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
# 3b Load BM25 index
# -----------------------------
def load_bm25_index(path):
    with open(path, "rb") as f:
        bm25_index = pickle.load(f)
    return bm25_index


# -----------------------------
# 4 Encode query
# -----------------------------
def encode_query(model, query):
    return model.encode([query])


# -----------------------------
# 5 Retrieve context (hybrid primary retrieval)
# -----------------------------
def retrieve_context(
    index,
    documents,
    query_embedding,
    query,
    bm25_index,
    k=3,
    dense_top_n=10,
    bm25_top_m=10,
    rrf_k0=60
):
    return hybrid_retrieve_context(
        index=index,
        documents=documents,
        bm25_index=bm25_index,
        query_embedding=query_embedding,
        query=query,
        dense_top_n=dense_top_n,
        bm25_top_m=bm25_top_m,
        k=k,
        rrf_k0=rrf_k0
    )


# -----------------------------
# 5b Vector search doc ids
# -----------------------------
def vector_search(index, query_embedding, top_n=10):
    _, indices = index.search(query_embedding, top_n)
    return [int(idx) for idx in indices[0] if idx != -1]


# -----------------------------
# 5c BM25 search doc ids
# -----------------------------
def tokenize_for_bm25(text):
    return re.findall(r"\w+", text.lower())


def bm25_search(bm25_index, query, top_m=10):
    query_tokens = tokenize_for_bm25(query)
    scores = bm25_index.get_scores(query_tokens)
    ranked_ids = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )
    return ranked_ids[:top_m]


# -----------------------------
# 5d Reciprocal Rank Fusion
# -----------------------------
def reciprocal_rank_fusion(ranked_lists, k0=60, top_k=3):
    fused_scores = {}

    for ranked_list in ranked_lists:
        for rank, doc_id in enumerate(ranked_list, start=1):
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + (1.0 / (k0 + rank))

    reranked_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)
    return reranked_ids[:top_k]


# -----------------------------
# 5e Hybrid retrieve context
# -----------------------------
def hybrid_retrieve_context(
    index,
    documents,
    bm25_index,
    query_embedding,
    query,
    dense_top_n=10,
    bm25_top_m=10,
    k=3,
    rrf_k0=60
):
    dense_doc_ids = vector_search(index, query_embedding, top_n=dense_top_n)
    bm25_doc_ids = bm25_search(bm25_index, query, top_m=bm25_top_m)

    fused_ids = reciprocal_rank_fusion(
        [dense_doc_ids, bm25_doc_ids],
        k0=rrf_k0,
        top_k=k
    )

    return [documents[idx] for idx in fused_ids]


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

    print("Loading BM25 index...")
    bm25_index = load_bm25_index(BM25_PATH)

    query = input("Enter your question: ")

    query_embedding = encode_query(embedding_model, query)

    contexts = retrieve_context(
        index=index,
        documents=documents,
        query_embedding=query_embedding,
        query=query,
        bm25_index=bm25_index,
        k=3
    )

    prompt = build_prompt(query, contexts)

    print("\nGenerating answer...\n")

    answer = generate_answer(prompt)

    print("Answer:")
    print(answer)


if __name__ == "__main__":
    main()