import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from rag_utils import (
    connect_db, ensure_tables, ingest_folder, search_chunks
)
from gemini_client import GeminiClient

st.set_page_config(page_title="Python Chatbot", page_icon="📂", layout="centered")
st.markdown("<h1 style='text-align: center;'>Python Chatbot</h1>", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

try:
    conn = connect_db()
    ensure_tables(conn)
    folder = os.getenv("DATA_FOLDER")
    if folder and os.path.isdir(folder):
        ingest_folder(conn, folder) 
    else:
        st.warning("No data folder found. Please set DATA_FOLDER in your .env file.")
except Exception as e:
    st.error(f"Ingestion failed: {e}")

st.divider()
st.markdown("<h3 style='text-align: center;'>💬 Ask any Question</h3>", unsafe_allow_html=True)

if not folder or not os.path.isdir(folder):
    st.info("Place files in the data folder to start chatting.")
else:
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_input = st.chat_input("Ask a question")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        try:
            conn = connect_db()
            results = search_chunks(conn, user_input, k=5)
        except Exception as e:
            with st.chat_message("assistant"):
                st.error(f"Search failed: {e}")
            st.stop()

        REFUSAL = (
            "Sorry, I am restricted to the ingested files and cannot answer that. "
            "Please ask something that can be answered from these documents."
        )

        if not results:
            response = REFUSAL
        else:
            best_dist = results[0][1]
            THRESH = 0.35 
            if best_dist > THRESH:
                response = REFUSAL
            else:
                context_blocks = [content for (content, _dist) in results]
                context = "\n\n".join(context_blocks)

                system_prompt = (
                    "You are a helpful assistant restricted to the provided context, "
                    "which comes from a set of ingested files (OCR-enabled). "
                    "Use the context as your only source. "
                    "If the answer is partially supported, explain what is present and what is missing. "
                    "Do not invent facts."
                )
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_input}"}
                ]
                try:
                    client = GeminiClient()
                    response = client.chat(messages, temperature=0.0, max_tokens=600)
                except Exception as e:
                    response = f"Gemini API error: {e}"

        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})