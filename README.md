# Custom Knowledge Base AI Chatbot Using GPT-3 Deployed on Streamlit

**Author**: Husani Newbold

**Date**: 2024-08-19

## Table of Contents
1. [Introduction & Project Description](#introduction--project-description)
2. [Code Walkthrough and Key Components](#code-walkthrough--key-components)
3. [How to Run the Project](#how-to-run-the-project)
4. [Improvements and Recommendations](#improvements-and-recommendations)
5. [Contributors](#contributors)


## Introduction & Project Description

### Introduction
This project leverages a GPT-based Large Language Model (LLM) from OpenAI to create a chatbot that can process text from a knowledge base built using a collection of custom PDF documents. The chatbot then uses generative AI to provide accurate answers to user queries based on this knowledge base. 

### Project Description
The chatbot application was built using Python and incorporates LangChain for document processing, Facebook AI Similarity Search (FAISS) for vector storage, and Streamlit to create an interactive web interface. Designed for easy extensibility, the application allows users to update the knowledge base by simply adding new PDF documents. For this particular project, a custom set of files was generated using ChatGPT, containing basic information about a fictional bank, "Bank XYZ," including details about locations, hours of operation, and products. Referencing these files, the bot application is able to accurately answer custom domain questions such as what types of products are offered, hours of operation for specific branches, or how to open a certain product.

<p align="center">
  <img src="Chatbot ScreenShot 1.png" alt="Image 1" width="300" height="200" style="margin-right: 50px;">
  <img src="Chatbot Screenshot 2.png" alt="Image 2" width="300" height="200">
</p>


## Code Walkthrough and Key Components

### 1. Environment Setup

```python
# Set environment variable to avoid OpenMP runtime error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load OpenAI API key from environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = openai_api_key
```
- The environment variable KMP_DUPLICATE_LIB_OK is set to TRUE to prevent conflicts that may arise due to different libraries using the OpenMP runtime. 
- The load_dotenv() function loads the API key from a .env file to keep it protected and out of the main script. 

### 2. PDF Processing
```python
# Convert all PDFs in the folder to text
pdf_folder = "./knowledge_base"  # Replace with the actual folder path containing your PDFs
docs = []

for filename in os.listdir(pdf_folder):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(pdf_folder, filename)
        text = textract.process(pdf_path).decode('utf-8')
        docs.append(text)

# Combine extracted text into one large document
combined_text = "\n".join(docs)

```
- This code iterates through all PDF files in a specified folder, extracts the text from each PDF using the textract library, and appends the extracted text to a list. The list is then combined into one large text block, which serves as the knowledge base for the chatbot to reference when answering user queries. 

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
- **GPT2TokenizerFast**: This tokenizer converts text into tokens, preparing it for processing by the model.

- **count_tokens Function**: This function counts the number of tokens in a given text, helping manage text length.

- **RecursiveCharacterTextSplitter**: This tool splits the text into smaller, manageable chunks based on token count, with some overlap between chunks to maintain context.

- **Chunk Creation**: The text is split into chunks that are easier for the model to search through when responding to queries.

### 4. Embeddings and FAISS Index Creation
```python
# Create embeddings and FAISS index
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(chunks, embeddings)
```
- The text chunks are transformed into vector embeddings using OpenAI embeddings. Vector embeddings are numerical representations of text that capture the semantic meaning and context of the content. These embeddings allow similar pieces of text to be located near each other in a high-dimensional space. These embeddings are then indexed using FAISS, which allows for efficient similarity search and is crucial for retrieving relevant documents when answering user queries.

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
- This function takes a user query, searches the FAISS index for relevant text chunks, and uses the GPT model to generate a response. The response is based on the most similar documents found in the knowledge base.

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
- This code sets up the Streamlit interface, allowing users to input queries and receive answers in an interactive, web-based format.

## How to Run the Project
### 1. Clone the Repository

```bash
git clone https://github.com/hnewbold/custom_knowledge_base_chatbot.git
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
Create a .env file in the root directory and add your own OpenAI API key:
```python
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Run the app using Streamlit
```python
streamlit run app.py
```
This will launch the chatbot application on a local web server where you can interact with the chatbot via the web interface.

## Improvements and Recommendations
- Adjust the GPT Model's Temperature Setting: Experiment with the temperature parameter, which controls the randomness of the model's output. Lowering the temperature makes the output more deterministic and focused, which might be better for specific domain questions. Conversely, slightly increasing it could make the responses more creative, which might be useful depending on the application's needs.

- Improve the Chatbot Interface: Enhance the user interface of the chatbot by adding features such as a conversation history pane and a more polished text input box. This would make the interaction more user-friendly, allowing users to follow the flow of the conversation and revisit previous responses without losing context.

- Optimize Text Storage and Processing: Consider implementing a more efficient way to store and process the text files from the knowledge base. This could involve using a database to store the documents and retrieve them more quickly or exploring more advanced text processing techniques to reduce memory usage and improve the overall performance of the application.

## Contributors
Husani Newbold (Author)

