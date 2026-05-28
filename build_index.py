import pandas as pd
import pickle
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.schema import Document


DATA_PATH = "data/rag_dataset.csv"
FAISS_STORE_PATH = "data/faiss_store"
BM25_PATH = "data/bm25_retriever.pkl"


# -----------------------------
# 1. Load dataset
# -----------------------------
def load_dataset(path):
    df = pd.read_csv(path)
    return df


# -----------------------------
# 2. Create LangChain Documents
# -----------------------------
# Trước đây: Tạo list of strings → lưu pickle
# Bây giờ: Tạo list of Document objects → LangChain yêu cầu format này
#           để tự động map vector ↔ nội dung text khi lưu vào FAISS
def create_langchain_documents(df):
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
        # Document object = page_content (nội dung) + metadata (thông tin phụ)
        doc = Document(
            page_content=text,
            metadata={
                "product_name": row["product_name"],
                "brand": row["brand"],
                "category": row["category"],
                "price": row["price_usd"]
            }
        )
        documents.append(doc)

    return documents


# -----------------------------
# 3. Load embedding model (LangChain wrapper)
# -----------------------------
# Trước đây: SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
# Bây giờ: HuggingFaceEmbeddings(model_name=...) — cùng model, nhưng bọc trong LangChain
def load_embedding_model():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    return embeddings


# -----------------------------
# 4. Build & Save FAISS index (LangChain)
# -----------------------------
# Trước đây: 3 bước thủ công (encode → IndexFlatL2 → add → write_index)
# Bây giờ: 1 dòng duy nhất — FAISS.from_documents() tự encode + build + lưu
def build_and_save_faiss(documents, embeddings):
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local(FAISS_STORE_PATH)
    return vector_store


# -----------------------------
# 5. Build & Save BM25 Retriever (LangChain)
# -----------------------------
# Trước đây: Tự tokenize bằng regex → BM25Okapi(tokenized_corpus) → pickle
# Bây giờ: BM25Retriever.from_documents() tự tokenize + build
def build_and_save_bm25(documents):
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 30  # Số lượng kết quả trả về khi tìm kiếm

    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25_retriever, f)

    return bm25_retriever


# -----------------------------
# Main pipeline
# -----------------------------
def main():

    print("Loading dataset...")
    df = load_dataset(DATA_PATH)

    print("Creating LangChain Documents...")
    documents = create_langchain_documents(df)

    print("Loading embedding model (HuggingFaceEmbeddings)...")
    embeddings = load_embedding_model()

    print("Building & saving FAISS vector store...")
    build_and_save_faiss(documents, embeddings)

    print("Building & saving BM25 retriever...")
    build_and_save_bm25(documents)

    print(f"\nIndex built successfully!")
    print(f"Total documents: {len(documents)}")
    print(f"FAISS store saved to: {FAISS_STORE_PATH}/")
    print(f"BM25 retriever saved to: {BM25_PATH}")


if __name__ == "__main__":
    main()