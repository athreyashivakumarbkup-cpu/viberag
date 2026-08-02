import streamlit as st
import os
from langchain_groq import ChatGroq

st.set_page_config(page_title="Employee Handbook Assistant", page_icon="📘")

st.title("📘 Employee Handbook Assistant")

with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    model = st.selectbox(
        "Model",
        ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
    )

if not api_key:
    st.info("Enter your Groq API key in the sidebar.")
    st.stop()

llm = ChatGroq(
    api_key=api_key,
    model=model,
    temperature=0
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask a question...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = llm.invoke(question).content
            st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
