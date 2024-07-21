import os
from dotenv import load_dotenv

from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms.octoai_endpoint import OctoAIEndpoint
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

file_path = "CustomerData.txt"

# Read the content of the file
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

embeddings = HuggingFaceEmbeddings()

vector_store = FAISS.from_documents(
    file_texts,
    embedding=embeddings
)

llm = OctoAIEndpoint(
    model="meta-llama-3-8b-instruct",
    max_tokens=1024,
    presence_penalty=0,
    temperature=0.1,
    top_p=0.9,
)

retriever = vector_store.as_retriever()

template = """Welcome to WolfTires! Your role as a salesperson is crucial for driving sales and maximizing conversions. 
Focus on increasing our call-to-sale conversion rate. Please answer customer queries as
a high performing salesperson and limit your responses to 2 sentences. Don't generate a whole conversation.
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

# Example query
#userInput = input()

#while(userInput != exit):
   # response = chain.invoke(userInput)
    #print(response)
    
while True:
    userInput = input()

    if userInput == 'exit':   
        break
    else:
        response = chain.invoke(userInput)
        print(response)

    if userInput == 'exit':
        break


