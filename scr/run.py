import fetch_datas
import encoding
import utils
import llm

def run_main(user_query, top_k):
    # load and process all data
    datas= fetch_datas.get_data(data_folder='data')

    # embedding data
    index, embedder, questions, answers= encoding.embedding(datas)

    # save vectorestore for reducing data loading time 
    utils.save_files(index, questions, answers)

    # load files
    # index, questions, answers= utils.load_files()

    result= llm.rag_query(user_query, top_k,  embedder, index, questions, answers)

    return result["response"]