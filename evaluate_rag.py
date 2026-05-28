import pandas as pd
from datasets import Dataset
import os
from dotenv import load_dotenv

# Cố gắng import RAGAS
try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
except ImportError as e:
    print(f"Lỗi Import RAGAS/LangChain: {e}. Vui lòng cài đặt đầy đủ theo requirements.txt")
    exit(1)

# Import hệ thống RAG (LangChain version)
# Trước đây: Import 12 hàm riêng lẻ
# Bây giờ: Chỉ cần 2 hàm chính
from rag_pipeline import load_rag_chain, ask

def main():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Lỗi: Vui lòng thêm GEMINI_API_KEY vào file .env")
        return
    # Bổ sung dòng này để LangChain nhận diện được API key
    os.environ["GOOGLE_API_KEY"] = api_key

    # ==========================================================
    # Bước 1: Nạp hệ thống RAG
    # ==========================================================
    # Trước đây: Load 5 thành phần riêng lẻ (embedding_model, faiss_index, docs, bm25, reranker)
    # Bây giờ: 1 dòng duy nhất — load_rag_chain() trả về chain hoàn chỉnh
    print("1. Đang nạp hệ thống RAG (LangChain Chain)...")
    try:
        chain = load_rag_chain()
    except Exception as e:
        print(f"Lỗi nạp hệ thống: {e}")
        return

    # ==========================================================
    # Bước 2: Đọc tập dữ liệu kiểm thử
    # ==========================================================
    print("2. Đang đọc tập dữ liệu kiểm thử (Testset)...")
    try:
        testset_df = pd.read_csv("data/ragas_testset.csv")
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy data/ragas_testset.csv. Vui lòng chạy generate_testset.py trước.")
        return

    questions = testset_df["question"].tolist()
    # Kiểm tra xem bộ test có cột ground_truth không
    if "ground_truth" in testset_df.columns:
        ground_truths = testset_df["ground_truth"].tolist()
    else:
        print("Cảnh báo: Không tìm thấy cột ground_truth, RAGAS sẽ không thể chấm context_recall.")
        ground_truths = [""] * len(questions)
    
    answers = []
    contexts_list = []

    # ==========================================================
    # Bước 3: Thu thập câu trả lời từ RAG
    # ==========================================================
    # Trước đây: 4 bước thủ công cho mỗi câu hỏi
    #   query_emb = encode_query(embedding_model, question)
    #   contexts = retrieve_context(index, docs, query_emb, question, bm25, reranker, k=3)
    #   prompt = build_prompt(question, contexts)
    #   answer = generate_answer(prompt)
    #
    # Bây giờ: 1 lệnh duy nhất — ask(chain, question)
    print("3. Đang thu thập câu trả lời từ RAG cho từng câu hỏi (Data Collection)...")
    for idx, question in enumerate(questions):
        print(f"  Đang xử lý câu {idx+1}/{len(questions)}...")
        
        result = ask(chain, question)
        
        # Lấy answer
        answer = result["answer"]
        
        # Lấy contexts (LangChain trả về list of Document objects)
        # RAGAS cần list of strings, nên phải convert
        contexts = [doc.page_content for doc in result["context"]]
        
        answers.append(answer)
        contexts_list.append(contexts)

    # ==========================================================
    # Bước 4: Chuẩn bị dữ liệu cho RAGAS
    # ==========================================================
    print("4. Chuẩn bị định dạng dữ liệu cho RAGAS...")
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths
    }
    dataset = Dataset.from_dict(data)

    # ==========================================================
    # Bước 5: Chấm điểm bằng RAGAS
    # ==========================================================
    print("5. Bắt đầu chấm điểm (Evaluation) bằng Giám khảo Gemini...")
    # Cấu hình Giám khảo
    evaluator_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)
    evaluator_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Gọi RAGAS evaluate
    try:
        from ragas.run_config import RunConfig
        # Buộc RAGAS chấm điểm từng câu một và tự động đợi nếu bị Google chặn
        run_config = RunConfig(max_workers=1, max_retries=100)

        result = evaluate(
            dataset=dataset,
            metrics=[
                context_precision,
                context_recall,
                faithfulness,
                answer_relevancy,
            ],
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
            run_config=run_config
        )
    except Exception as e:
        print(f"\n[!] Lỗi khi đánh giá: {e}")
        return

    print("\n================== BẢNG ĐIỂM TỔNG KẾT ==================")
    print(result)
    print("========================================================")

    # Lưu chi tiết điểm của từng câu hỏi để phân tích sau
    result_df = result.to_pandas()
    output_path = "data/evaluation_results.csv"
    result_df.to_csv(output_path, index=False)
    print(f"\n✅ Đã lưu bảng điểm chi tiết của từng câu hỏi tại: {output_path}")

if __name__ == "__main__":
    main()
