# Custom Knowledge Base Chatbot

## Introduction

Welcome to the Custom Knowledge Base Chatbot project! This project is a chatbot application that uses a Large Language Model (LLM) to process text from a collection of PDF documents and a GPT model (via OpenAI) to provide answers to user queries. The application is built using Python, leveraging LangChain, FAISS for vector storage, and Streamlit to create an interactive web interface. The chatbot is designed to be easily extensible, allowing users to update the knowledge base by simply adding new PDF documents.

## Key Features
- **Custom Knowledge Base:** The chatbot processes PDF documents, extracting text to create a searchable knowledge base.
- **Advanced Query Handling:** The chatbot uses a pre-trained GPT model to understand and answer user queries based on the knowledge base.
- **Streamlit Integration:** The chatbot is hosted on a Streamlit interface, providing a simple and intuitive way to interact with the model.

## Key Parts of the Code

### 1. Environment Setup

```python
# Set environment variable to avoid OpenMP runtime error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load environment variables from .env file (including OpenAI API key)
load_dotenv()
```
Purpose: The environment variable is set to avoid conflicts related to OpenMP runtime, which may arise due to different libraries using OpenMP. The load_dotenv() function loads environment variables from a .env file, ensuring that sensitive information like API keys is not hardcoded.

```python
# Step 1: Convert all PDFs in the folder to text
pdf_folder = "./knowledge_base"  # Replace with the actual folder path containing your PDFs
docs = []
```

### 2. PDF Processing
for filename in os.listdir(pdf_folder):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(pdf_folder, filename)
        text = textract.process(pdf_path).decode('utf-8')
        docs.append(text)
```
Purpose: This code iterates through all PDF files in a specified folder, extracts the text from each PDF using the textract library, and appends the extracted text to a list. This list is then combined into one large text block to be used as the knowledge base.

### 3. Text Splitting and Tokenization
```python
# Tokenize and split the text into chunks
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=24,
    length_function=count_tokens,
)

chunks = text_splitter.create_documents([combined_text])
```
Purpose: The text from the PDFs is tokenized and split into manageable chunks using a recursive character splitter. This helps in creating smaller, indexed pieces of text that can be efficiently searched when responding to queries.

### 4. Embeddings and FAISS Index Creation
```python
# Create embeddings and FAISS index
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(chunks, embeddings)
```
Purpose: The text chunks are transformed into vector embeddings using OpenAI embeddings. These embeddings are then indexed using FAISS, which allows for efficient similarity search and is crucial for retrieving relevant documents when answering user queries.

### 5. Query Handling
```python
# Define a function to answer queries
def answer_query(query, db):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5)
    chain = load_qa_chain(llm, chain_type="stuff")
    docs = db.similarity_search(query)
    response = chain.run(input_documents=docs, question=query)
    return response
```
Purpose: This function takes a user query, searches the FAISS index for relevant text chunks, and uses the GPT model to generate a response. The response is based on the most similar documents found in the knowledge base.

### 6. Streamlit Interface
```python
# Streamlit app setup
st.title("Custom Knowledge Base Chatbot")

# Ask questions
query = st.text_input("Enter your query:")
if st.button("Get Answer"):
    if query:
        answer = answer_query(query, db)
        st.write(f"**You:** {query}")
        st.write(f"**Bot:** {answer}")
    else:
        st.write("Please enter a query.")
```
Purpose: This code sets up the Streamlit interface, allowing users to input queries and receive answers in an interactive, web-based format.

## How to Run the Project
### 1. Clone the Repository
```python
git clone https://github.com/your-username/custom_knowledge_base_chatbot.git
cd custom_knowledge_base_chatbot
```
### 2. Set Up the Environment
#### a) Install the required packages
Create a virtual environment and install the required Python packages using requirements.txt
```python
python -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`
pip install -r requirements.txt
```
#### b) Set up OpenAI API Key
Create a .env file in the root directory and add your OpenAI API key:
```python
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Run the app using Streamlit
```python
streamlit run app.py
```
This will start the chatbot on a local web server. You can interact with the chatbot via the web interface.

