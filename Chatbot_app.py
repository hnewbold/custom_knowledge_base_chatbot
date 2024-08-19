# Load environment variables from the .env file
from dotenv import load_dotenv
load_dotenv()

# Import necessary modules and libraries
import os
import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from transformers import GPT2TokenizerFast
from langchain.text_splitter import RecursiveCharacterTextSplitter
import textract

# Set environment variable to avoid OpenMP runtime error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OpenAI API key is not set. Please check your .env file.")
    st.stop()

os.environ["OPENAI_API_KEY"] = openai_api_key

# Convert and store all PDFs from Knowledge Base folder as text
pdf_folder = "./knowledge_base" 
docs = []

for filename in os.listdir(pdf_folder):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(pdf_folder, filename)
        text = textract.process(pdf_path).decode('utf-8')
        docs.append(text)

# Combine extracted text into one large document
combined_text = "\n".join(docs)

## Tokenize and split the text into chunks

# Load the GPT-2 tokenizer from the Hugging Face
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# Define a function to count the number of tokens in a given text.
def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

# Initialize a RecursiveCharacterTextSplitter to split the text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,            
    chunk_overlap=24,          
    length_function=count_tokens, 
)

# Split the combined text into smaller chunks using the text splitter.
chunks = text_splitter.create_documents([combined_text])

## Create embeddings and FAISS index

# Initialize the embeddings object to convert text chunks into vector embeddings
embeddings = OpenAIEmbeddings()
# Use FAISS (Facebook AI Similarity Search) to create an index from the document chunks
db = FAISS.from_documents(chunks, embeddings)

# Define a function to answer queries
def answer_query(query, db):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5)
    chain = load_qa_chain(llm, chain_type="stuff")
    docs = db.similarity_search(query)
    response = chain.run(input_documents=docs, question=query)
    return response

## Streamlit app setup

# Set the title of the Streamlit app
st.title("Custom Knowledge Base Chatbot")

# Create an input text box for the user to enter their query
query = st.text_input("Enter your query:")

# Create a button that triggers the chatbot's response when clicked
if st.button("Get Answer"):
    if query:
        answer = answer_query(query, db)
        st.write(f"**You:** {query}")
        st.write(f"**Bot:** {answer}")
    else:
        st.write("Please enter a query.")

