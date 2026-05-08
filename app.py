import streamlit as st
import os

# Import functions from existing pipeline
from rag_pipeline import (
    INDEX_PATH,
    DOC_PATH,
    load_embedding_model,
    load_index,
    load_documents,
    encode_query,
    retrieve_context,
    build_prompt,
    generate_answer
)

# Streamlit App Configuration
st.set_page_config(page_title="E-commerce Assistant", page_icon="🛒", layout="centered")
st.title("🛒 E-commerce Shopping Assistant")
st.markdown("Xin chào! Tôi là trợ lý AI của cửa hàng. Bạn cần tìm sản phẩm gì hôm nay?")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load Resources (Cached so it only runs once)
@st.cache_resource(show_spinner=False)
def load_resources():
    with st.spinner("Đang tải dữ liệu cửa hàng và AI... Vui lòng đợi nhé."):
        model = load_embedding_model()
        index = load_index(INDEX_PATH)
        documents = load_documents(DOC_PATH)
    return model, index, documents

try:
    embedding_model, faiss_index, docs = load_resources()
except Exception as e:
    st.error(f"Lỗi tải dữ liệu: {e}")
    st.stop()

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("VD: Recommend a blue cotton shirt for summer"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Retrieve and Generate
    with st.spinner("Đang tìm kiếm sản phẩm và phản hồi..."):
        try:
            # 1. Encode query
            query_embedding = encode_query(embedding_model, prompt)
            
            # 2. Retrieve context
            contexts = retrieve_context(faiss_index, docs, query_embedding, k=3)
            
            # 3. Build Prompt
            llm_prompt = build_prompt(prompt, contexts)
            
            # 4. Generate Answer
            answer = generate_answer(llm_prompt)
            
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(answer)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except Exception as e:
            st.error(f"Đã có lỗi xảy ra: {e}")
