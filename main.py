MISTRAL_API_KEY = "UCQISGNexbhV9n3bEfmu1U0M7pYpC2Xj"

import pandas as pd
import numpy as np
import faiss
import requests
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load index
index = faiss.read_index("faiss_index.bin")

# Load questions and answers
import pickle
with open("questions.pkl", "rb") as f:
    questions = pickle.load(f)
with open("answers.pkl", "rb") as f:
    answers = pickle.load(f)


def retrieve(user_query, top_k=2):
    # Encode query
    query_vec = embedder.encode([user_query], convert_to_numpy=True)
    faiss.normalize_L2(query_vec)
    
    # Search in FAISS
    D, I = index.search(query_vec, top_k)
    
    results = []
    for idx, score in zip(I[0], D[0]):
        results.append({
            "question": questions[idx],
            "answer": answers[idx],
            "score": float(score)
        })
    return results


MISTRAL_MODEL = "open-mixtral-8x22b"
MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"

def rag_query(user_query, top_k=3):
    retrieved = retrieve(user_query, top_k)

    prompt = f"""
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

"""

    payload = {
        "model": MISTRAL_MODEL,
        "messages": [{"role": "user", "content": prompt}]
    }
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    response = requests.post(MISTRAL_URL, headers=headers, json=payload)
    response_json = response.json()

    # Extract text from API response
    answer = response_json["choices"][0]["message"]["content"]
    return {"query": user_query, "retrieved": retrieved, "response": answer}

answer= rag_query(user_query='im happy', top_k=2)
print(answer)