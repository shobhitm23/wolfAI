from fastapi import FastAPI, Security, HTTPException
import os
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain_community.embeddings import OctoAIEmbeddings
from langchain_community.llms.octoai_endpoint import OctoAIEndpoint
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from pprint import pprint


load_dotenv()

# OCTOAI_API_TOKEN = os.getenv("OCTO_AI_TOKEN")
from langchain_text_splitters import RecursiveCharacterTextSplitter, HTMLHeaderTextSplitter

url = "https://en.wikipedia.org/wiki/Star_Wars"

headers_to_split_on = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
    ("h3", "Header 3"),
    ("h4", "Header 4"),
    ("div", "Divider")
]

html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

# for local file use html_splitter.split_text_from_file(<path_to_file>)
html_header_splits = html_splitter.split_text_from_url(url)

chunk_size = 1024
chunk_overlap = 128
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
)

# Split
splits = text_splitter.split_documents(html_header_splits)

llm = OctoAIEndpoint(
        model="llama-2-13b-chat-fp16",
        max_tokens=1024,
        presence_penalty=0,
        temperature=0.1,
        top_p=0.9
    )
embeddings = OctoAIEmbeddings(endpoint_url="https://text.octoai.run/v1/")

vector_store = FAISS.from_documents(
    splits,
    embedding=embeddings
)

retriever = vector_store.as_retriever()

template="""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:"""
prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

chain.invoke("Who is Luke's Father?")

template="""You are a literary critic. You are given some context and asked to answer questions based on only that context.
Question: {question} 
Context: {context} 
Answer:"""
lit_crit_prompt = ChatPromptTemplate.from_template(template)

lcchain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | lit_crit_prompt
    | llm
    | StrOutputParser()
)

pprint(lcchain.invoke("What is the best thing about Luke's Father's story line?"))
