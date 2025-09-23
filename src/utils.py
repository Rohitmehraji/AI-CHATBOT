from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


def file_loader(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents


def split_doc():
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    return splitter


def embedding(model_name):
    embeddings_model = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings_model