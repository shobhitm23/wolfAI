import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.llms.octoai_endpoint import OctoAIEndpoint
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Define data model for incoming queries
class QueryModel(BaseModel):
    customer: str

# File path and read content
file_path = "CustomerData.txt"

with open(file_path, 'r') as f:
    file_text = f.read()

# Initialize the text splitter
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=512, chunk_overlap=64
)

# Split the text into chunks
texts = text_splitter.split_text(file_text)

# Create a list of Document objects for the split texts
file_texts = [
    Document(page_content=chunked_text, 
             metadata={"doc_title": os.path.basename(file_path).split(".")[0], 
                       "chunk_num": i})
    for i, chunked_text in enumerate(texts)
]

# Initialize embeddings and vector store
embeddings = HuggingFaceEmbeddings()

vector_store = FAISS.from_documents(
    file_texts,
    embedding=embeddings
)

# Initialize LLM endpoint
llm = OctoAIEndpoint(
    model="meta-llama-3-8b-instruct",
    max_tokens=1024,
    presence_penalty=0,
    temperature=0.1,
    top_p=0.9,
)

# Initialize retriever
retriever = vector_store.as_retriever()

# Define the prompt template
template = """Welcome to WolfTires! Your role as a salesperson is crucial for driving sales and maximizing conversions. 
Focus on increasing our call-to-sale conversion rate. Please answer customer queries as
a high performing salesperson and limit your responses to 2 sentences. Don't generate a whole conversation.

Customer: {customer} 
Context: {context} 
Answer:"""

prompt = ChatPromptTemplate.from_template(template)

# Create the chain
chain = (
    {"context": retriever, "customer": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Define the FastAPI endpoint
@app.post("/ask/")
def ask_question(query: QueryModel):
    try:
        response = chain.invoke(query.customer)
        return {"WolfAI": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
