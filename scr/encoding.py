from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


def embedding(all_datas):

    # Load embedding model
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # Extract questions (keys) and answers (values)
    questions = [str(q) for q, a in all_datas if q is not None and str(q).strip() != ""]
    answers = [str(a) for q, a in all_datas if a is not None and str(a).strip() != ""]

    # Encode questions into vectors
    question_embeddings = embedder.encode(questions, convert_to_numpy=True)

    # Create FAISS index (cosine similarity)
    d = question_embeddings.shape[1]  # embedding dimension
    index = faiss.IndexFlatIP(d)  # inner product = cosine similarity if normalized

    # Normalize for cosine similarity
    faiss.normalize_L2(question_embeddings)
    index.add(question_embeddings)
    return  index, embedder, questions, answers