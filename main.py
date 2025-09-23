import streamlit as st
from src.chain import rag_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from src.prompt import system_prompt
from langchain_core.prompts import ChatPromptTemplate
from src.vector_store import vector_store_retriver
from src.utils import file_loader, split_doc, embedding
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

pages = file_loader("research_paper.pdf")  # Location of Documents you want to load.For example : We are loading Research Paper you can load any other documents

splitter = split_doc()

docs = splitter.split_documents(pages)

embeddings_model = embedding(model_name="sentence-transformers/all-MiniLM-L6-v2") # Hugging Face Embedding Model 

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

parser = StrOutputParser()

prompt = ChatPromptTemplate.from_template(system_prompt)

retriever = vector_store_retriver(docs, embeddings_model)

chain = rag_chain(llm, parser, prompt, retriever)

st.title("Custom Chatbot")

# Display initial AI message
with st.chat_message("ai"):
    st.write("Hello 👋, How can I assist you today?")

# Get user input
user_input = st.chat_input("Enter your query")

if user_input:
    # Display user's message
    with st.chat_message("human"):
        st.write(user_input)

    # Get response from your chain
    response = chain.invoke(user_input)

    # Display AI's response
    with st.chat_message("ai"):
        st.write(response)
