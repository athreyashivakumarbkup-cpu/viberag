import streamlit as st
import os

from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

st.set_page_config(page_title="Employee Handbook Assistant", page_icon="📘")

st.title("📘 Employee Handbook Assistant")
st.write("Ask questions about the employee handbook.")

# -----------------------------
# API Key
# -----------------------------
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
)

# -----------------------------
# Paste your CLASSIFY_PROMPT
# -----------------------------

CLASSIFY_PROMPT = """PASTE YOUR PROMPT HERE"""

def classify_question(question):
    prompt = CLASSIFY_PROMPT.format(question=question)
    response = llm.invoke(prompt).content.strip().lower()
    return "policy" if "policy" in response else "general"

# -----------------------------
# Paste handbook_text
# -----------------------------

handbook_text = """PASTE YOUR HANDBOOK"""

handbook_chunks = [
    chunk.strip()
    for chunk in handbook_text.split("\n\n")
    if chunk.strip()
]

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_texts(
    handbook_chunks,
    embedding_model,
)

def retrieve(query, k=2):
    docs = vectorstore.similarity_search(query, k=k)
    return [d.page_content for d in docs]

class HandbookState(TypedDict):
    question: str
    category: str
    context: str
    answer: str

def router_node(state):
    return {"category": classify_question(state["question"])}

def rag_node(state):
    chunks = retrieve(state["question"])
    context = "\n".join(chunks)

    prompt = f"""
Answer ONLY from the context.

Context:
{context}

Question:
{state['question']}

Answer:
"""

    answer = llm.invoke(prompt).content

    return {
        "context": context,
        "answer": answer
    }

def general_node(state):
    answer = llm.invoke(state["question"]).content

    return {
        "context": "",
        "answer": answer
    }

def route(state):
    if state["category"] == "policy":
        return "rag_node"
    return "general_node"

builder = StateGraph(HandbookState)

builder.add_node("router", router_node)
builder.add_node("rag_node", rag_node)
builder.add_node("general_node", general_node)

builder.add_edge(START, "router")

builder.add_conditional_edges(
    "router",
    route,
    ["rag_node", "general_node"]
)

builder.add_edge("rag_node", END)
builder.add_edge("general_node", END)

agent = builder.compile()

# -----------------------------
# Chat UI
# -----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask a question...")

if question:

    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.spinner("Thinking..."):
        result = agent.invoke({"question": question})
        answer = result["answer"]

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )