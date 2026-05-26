import os
import pickle
import pandas as pd
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

# Cố gắng import RAGAS, tương thích với RAGAS v0.2.x
try:
    from ragas.testset import TestsetGenerator
except ImportError as e:
    print(f"Lỗi Import RAGAS: {e}. Vui lòng đảm bảo bạn đã cài đặt các thư viện trong yêu cầu mới.")
    exit(1)

def main():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Lỗi: Vui lòng thêm GEMINI_API_KEY vào file .env")
        return
    
    # Bổ sung dòng này để LangChain nhận diện được API key của bạn
    os.environ["GOOGLE_API_KEY"] = api_key

    print("1. Đang nạp dữ liệu sản phẩm từ data/documents.pkl...")
    try:
        with open("data/documents.pkl", "rb") as f:
            raw_documents = pickle.load(f)
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file data/documents.pkl. Vui lòng chạy build_index.py trước.")
        return

    # Gộp tất cả sản phẩm thành một chuỗi lớn để RAGAS xử lý (Tránh lỗi tài liệu quá ngắn)
    # RAGAS 0.2.x yêu cầu document phải dài hơn 100 token để có thể phân tích Knowledge Graph
    combined_text = "\n\n---\n\n".join(raw_documents)
    documents = [Document(page_content=combined_text)]

    print("2. Đang khởi tạo Giám khảo sinh đề (Gemini)...")
    # Khởi tạo mô hình ngôn ngữ và mô hình nhúng của Google
    # Dùng gemini-2.5-flash để tiết kiệm tốc độ và chi phí
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Khởi tạo TestsetGenerator (RAGAS v0.2.x)
    generator = TestsetGenerator.from_langchain(
        llm=llm,
        embedding_model=embeddings
    )

    print("3. Bắt đầu sinh 15 câu hỏi tự động bằng RAGAS (quá trình này có thể mất vài phút)...")
    try:
        from ragas.run_config import RunConfig
        # Giới hạn 1 luồng duy nhất và cấu hình retry tối đa 100 lần
        # để tránh lỗi Rate Limit khi dùng Free Tier của Google API
        run_config = RunConfig(max_workers=1, max_retries=100)

        # Tương thích RAGAS 0.2.x, sinh tự động dựa trên knowledge graph ngẫu nhiên
        testset = generator.generate_with_langchain_docs(
            documents,
            testset_size=15,
            run_config=run_config
        )
    except Exception as e:
        print(f"\n[!] Lỗi khi sinh dữ liệu: {e}")
        print("Gợi ý: Thường là do API Rate Limit của Gemini. Bạn có thể giảm testset_size xuống 5 để thử lại.")
        return

    print("4. Chuyển đổi và lưu kết quả...")
    # Lưu kết quả thành Pandas DataFrame rồi xuất ra CSV
    df = testset.to_pandas()
    output_path = "data/ragas_testset.csv"
    df.to_csv(output_path, index=False)
    
    print(f"\n✅ Hoàn tất! Đã lưu testset thành công tại: {output_path}")
    print("Bạn có thể mở file CSV này lên để xem thử các câu hỏi do AI đẻ ra.")

if __name__ == "__main__":
    main()
