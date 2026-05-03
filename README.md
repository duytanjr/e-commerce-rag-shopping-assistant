# 🛒 E-commerce RAG Shopping Assistant

An intelligent shopping assistant pipeline built with **Retrieval-Augmented Generation (RAG)**. This project retrieves relevant product information from a custom catalog and uses a Large Language Model (LLM) to generate personalized, format-specific recommendations for users.

## 🌟 Key Features

* **Vector Search Engine:** Utilizes `FAISS` for fast and efficient similarity search.
* **Multilingual Embeddings:** Powered by `sentence-transformers` (`paraphrase-multilingual-MiniLM-L12-v2`) to handle diverse queries accurately.
* **Generative AI Integration:** Uses Google's `Gemini` API to synthesize retrieved context into natural, helpful, and strictly formatted recommendations.
* **Secure Configuration:** Implements `python-dotenv` for safe API key management.
* **Modular Architecture:** Clean separation of concerns between data indexing (offline) and querying (online).

## 📂 Project Structure

```text
📦 RAG-Shopping-Assistant
 ┣ 📂 data/
 ┃ ┣ 📜 rag_dataset.csv        # Raw product catalog dataset
 ┃ ┣ 📜 product_index.faiss    # Compiled FAISS vector index
 ┃ ┗ 📜 documents.pkl          # Pickled document store for retrieval
 ┣ 📜 build_index.py           # Script to encode text and build FAISS index
 ┣ 📜 query.py                 # Script to test vector retrieval directly
 ┣ 📜 rag_pipeline.py          # Main pipeline integrating FAISS and Gemini LLM
 ┣ 📜 .env                     # Environment variables (API Keys) - NOT tracked by git
 ┣ 📜 .gitignore               # Git ignore rules
 ┗ 📜 README.md                # Project documentation
```
## 💡 Usage

Step 1: Build the Vector Index
Run the indexing script to process the dataset, generate embeddings, and save the FAISS index.
python build_index.py

Step 2: Test Retrieval (Optional)
You can test the similarity search directly without invoking the LLM to verify the embedding quality.
python query.py

Step 3: Run the Full RAG Pipeline
Interact with the shopping assistant. The system will retrieve the top 3 most relevant products and use Gemini to formulate the answer.
python rag_pipeline.py

📝 Example Output
Enter your question: Recommend a blue cotton shirt for summer.

Generating answer...

Answer:
1. MekongBasics Blue Shirt - Cotton - $73.8
   This blue cotton shirt is designed for everyday comfort, making it a great choice for summer.

## 👨‍💻 Author
Nguyễn Duy Tân
- Computer Science Major
- Passionate about AI, LLMs, and Data Science.