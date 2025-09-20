import pandas as pd
import numpy as np
import faiss
import requests
from sentence_transformers import SentenceTransformer
import pickle
from fastapi import FastAPI, Query as FQuery
from pydantic import BaseModel

# -----------------------------
# Load embedding model
# -----------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index
index = faiss.read_index("faiss_index.bin")

# Load Q&A
with open("questions.pkl", "rb") as f:
    questions = pickle.load(f)
with open("answers.pkl", "rb") as f:
    answers = pickle.load(f)

# -----------------------------
# FAISS retrieval function
# -----------------------------
def retrieve(user_query, top_k=2):
    query_vec = embedder.encode([user_query], convert_to_numpy=True)
    faiss.normalize_L2(query_vec)
    D, I = index.search(query_vec, top_k)
    
    results = []
    for idx, score in zip(I[0], D[0]):
        results.append({
            "question": questions[idx],
            "answer": answers[idx],
            "score": float(score)
        })
    return results

# -----------------------------
# RAG chat with per-user history
# -----------------------------
MISTRAL_API_KEY = "your_api_ley"
MISTRAL_MODEL = "open-mixtral-8x22b"
MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"

# Store chat history for each user_id
user_histories = {}

def rag_chat(user_id, user_query, top_k=3):
    chat_history = user_histories.get(user_id, [])

    # Retrieve context
    retrieved = retrieve(user_query, top_k)

    # Build messages including previous chat history
    messages = []
    for chat in chat_history:
        messages.append({"role": "user", "content": chat["user"]})
        messages.append({"role": "assistant", "content": chat["assistant"]})

    # Add retrieved context
    context_text = ""
    for i, item in enumerate(retrieved):
        context_text += f"{i+1}. Q: {item['question']}\n   A: {item['answer']}\n"

    messages.append({"role": "user", "content": f"""
You are a compassionate and helpful mental health assistant. 
Use the following context to answer the user's question as supportively as possible. 
Do not give medical advice, but provide understanding, comfort, and general coping suggestions. 

Context:
1. Q: I feel sad.
   A: I hear you. Want to talk about it?

2. Q: You seem sad.
   A: Just a little.

3. Q: You seem sad.
   A: Just a little.

User's question: im sad

Answer in a kind, empathetic, and supportive way.
"""})

    # Call Mistral API
    headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": MISTRAL_MODEL, "messages": messages}
    response = requests.post(MISTRAL_URL, headers=headers, json=payload)
    response_json = response.json()
    answer = response_json["choices"][0]["message"]["content"]

    # Update user's chat history
    chat_history.append({"user": user_query, "assistant": answer})
    user_histories[user_id] = chat_history

    return {"query": user_query, "retrieved": retrieved, "response": answer}

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Mental Health RAG Chat API")

# Root endpoint
@app.get("/")
def root():
    return {"message": "Mental Health RAG Chat API is running. Use /chat endpoint."}

# Pydantic model for POST requests
class Query(BaseModel):
    user_id: str
    question: str
    top_k: int = 2

# POST endpoint
@app.post("/chat")
def chat_post(query: Query):
    return rag_chat(query.user_id, query.question, top_k=query.top_k)

# GET endpoint for quick testing in browser
@app.get("/chat")
def chat_get(
    user_id: str = FQuery(...),
    question: str = FQuery(...),
    top_k: int = FQuery(2)
):
    return rag_chat(user_id, question, top_k=top_k)

