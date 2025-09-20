import faiss


def save_files(index, questions, answers):
        # Save index
    faiss.write_index(index, "fiass_index_/faiss_index.bin")

    # Optionally, save the questions/answers separately
    import pickle
    with open("fiass_index_/questions.pkl", "wb") as f:
        pickle.dump(questions, f)
    with open("fiass_index_/answers.pkl", "wb") as f:
        pickle.dump(answers, f)


def load_files():
    # Load index
    index = faiss.read_index("fiass_index_/faiss_index.bin")

    # Load questions and answers
    import pickle
    with open("fiass_index_/questions.pkl", "rb") as f:
        questions = pickle.load(f)
    with open("fiass_index_/answers.pkl", "rb") as f:
        answers = pickle.load(f)

    return index, questions, answers
