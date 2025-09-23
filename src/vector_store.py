from langchain_community.vectorstores import FAISS


def vector_store_retriver(docs, embeddings_model):
    vectorstore = FAISS.from_documents(docs, embeddings_model)
    retriever = vectorstore.as_retriever()
    return retriever

