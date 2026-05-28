import os
import pickle
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain


FAISS_STORE_PATH = "data/faiss_store"
BM25_PATH = "data/bm25_retriever.pkl"


# =============================================================
# 1. Load Embedding Model (LangChain wrapper)
# =============================================================
# Trước đây: SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
# Bây giờ: HuggingFaceEmbeddings(model_name=...) — cùng model bên trong
def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )


# =============================================================
# 2. Load FAISS Vector Store
# =============================================================
# Trước đây: faiss.read_index("product_index.faiss") + pickle.load("documents.pkl")
#            → Phải tự map ID → document text thủ công
# Bây giờ: FAISS.load_local() tự động load cả vector lẫn text
def load_faiss_store(embeddings):
    return FAISS.load_local(
        FAISS_STORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )


# =============================================================
# 3. Load BM25 Retriever
# =============================================================
# Trước đây: pickle.load() → BM25Okapi object (phải tự tokenize query, tự tính score)
# Bây giờ: pickle.load() → BM25Retriever object (gọi .invoke(query) là xong)
def load_bm25_retriever():
    with open(BM25_PATH, "rb") as f:
        return pickle.load(f)


# =============================================================
# 4. Build Hybrid Retriever (FAISS + BM25 + RRF + Reranker)
# =============================================================
# Trước đây: 5 hàm thủ công (vector_search, bm25_search,
#            reciprocal_rank_fusion, hybrid_retrieve_context, rerank_documents)
#            tổng cộng ~120 dòng code
# Bây giờ: 3 class LangChain xếp chồng lên nhau, ~15 dòng code
def build_retriever(faiss_store, bm25_retriever):
    # 4a. Tạo FAISS retriever từ vector store
    faiss_retriever = faiss_store.as_retriever(search_kwargs={"k": 30})

    # 4b. Cấu hình BM25 retriever
    bm25_retriever.k = 30

    # 4c. Hybrid Search: EnsembleRetriever tự động chạy RRF bên trong
    # Trước đây ta phải tự viết công thức: score += 1/(k0 + rank)
    # Bây giờ LangChain đã code sẵn thuật toán RRF trong class này
    ensemble_retriever = EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever],
        weights=[0.5, 0.5]  # Trọng số ngang nhau cho FAISS và BM25
    )

    # 4d. Cross-Encoder Reranker: Chấm điểm lại Top-15 → chọn Top-3
    # Trước đây ta phải tự loop qua Cross-Encoder, tự sort, tự slice
    # Bây giờ LangChain gói gọn trong ContextualCompressionRetriever
    reranker_model = HuggingFaceCrossEncoder(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    compressor = CrossEncoderReranker(model=reranker_model, top_n=3)

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever
    )

    return compression_retriever


# =============================================================
# 5. Build RAG Chain (Retriever + Prompt + LLM)
# =============================================================
# Trước đây: 3 hàm riêng lẻ (build_prompt, generate_answer, retrieve_context)
#            phải gọi tuần tự thủ công
# Bây giờ: 1 chain duy nhất — ném câu hỏi vào → tự retrieve → tự prompt → tự generate
def build_rag_chain(retriever):
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY", "").strip('"').strip("'")
    os.environ["GOOGLE_API_KEY"] = api_key

    # 5a. Cấu hình LLM (giữ nguyên Gemini 2.5 Flash)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.7,
        google_api_key=api_key
    )

    # 5b. Prompt Template (giữ nguyên nội dung prompt cũ)
    # Trước đây: Nối chuỗi f-string thủ công
    # Bây giờ: ChatPromptTemplate — chuẩn LangChain, dễ tái sử dụng
    prompt = ChatPromptTemplate.from_template("""
You are a helpful shopping assistant.

Use ONLY the information from the product catalog below.

Context:
{context}

User question:
{input}

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
""")

    # 5c. Tạo chain hoàn chỉnh
    # create_stuff_documents_chain: Ghép tất cả documents vào {context} trong prompt
    # create_retrieval_chain: Kết nối retriever với document chain
    qa_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, qa_chain)

    return rag_chain


# =============================================================
# 6. Helper functions (cho app.py và evaluate_rag.py)
# =============================================================
def load_rag_chain():
    """Load toàn bộ hệ thống RAG và trả về chain hoàn chỉnh."""
    embeddings = load_embedding_model()
    faiss_store = load_faiss_store(embeddings)
    bm25_retriever = load_bm25_retriever()
    retriever = build_retriever(faiss_store, bm25_retriever)
    chain = build_rag_chain(retriever)
    return chain


def load_retriever():
    """Load chỉ phần retriever (để evaluate_rag.py lấy contexts riêng)."""
    embeddings = load_embedding_model()
    faiss_store = load_faiss_store(embeddings)
    bm25_retriever = load_bm25_retriever()
    return build_retriever(faiss_store, bm25_retriever)


def ask(chain, query):
    """
    Hỏi chatbot 1 câu và nhận kết quả.

    Trả về dict:
      - "answer": Câu trả lời từ Gemini
      - "context": Danh sách Document objects đã retrieve
    """
    result = chain.invoke({"input": query})
    return result


# =============================================================
# Main pipeline (chạy thử trực tiếp)
# =============================================================
def main():
    print("Loading RAG chain...")
    chain = load_rag_chain()

    query = input("\nEnter your question: ")

    print("\nProcessing...\n")
    result = ask(chain, query)

    print("Answer:")
    print(result["answer"])


if __name__ == "__main__":
    main()