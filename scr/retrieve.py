import faiss

def retrieve_data(user_query, embedder, index, questions, answers, top_k=2):
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