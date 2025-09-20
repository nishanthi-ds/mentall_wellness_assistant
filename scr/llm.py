import os
import requests
import retrieve

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_MODEL = "open-mixtral-8x22b"
MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"

prompt=  f"""
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


def rag_query(user_query, top_k,  embedder, index, questions, answers):

    retrieved = retrieve(user_query, top_k, embedder, index, questions, answers)
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

