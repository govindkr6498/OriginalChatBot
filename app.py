from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS  
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from fastapi import FastAPI
from PyPDF2 import PdfReader
import os
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PDF_PATH = "/home/ubuntu/ApexDeveloperGuidea.pdf"
# Load OpenAI API Key
if not OPENAI_API_KEY:
    raise ValueError("ERROR: OPENAI_API_KEY is missing from .env file!")

# Load PDF and create vector store
def get_vectorstore_from_static_pdf(pdf_path=PDF_PATH):
    pdf_reader = PdfReader(pdf_path)
    text = "".join([page.extract_text() or "" for page in pdf_reader.pages])

    # Split text into chunks
    from langchain.text_splitter import CharacterTextSplitter
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)

    # Create a vectorstore from the chunks
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(chunks, embeddings)

    return vector_store

# Load vector store at startup
vector_store = get_vectorstore_from_static_pdf()

# Chat request model
class ChatRequest(BaseModel):
    message: str

# Store chat history
chat_history = []

# Function to get response based on chat history
def get_response(user_input):
    global chat_history

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    documents = retriever.get_relevant_documents(user_input)

    if not documents:  
        return "I don't know. The PDF does not contain relevant information."

    # Create conversation-aware response
    context = "\n".join([doc.page_content for doc in documents])
    chat_history.append(HumanMessage(content=user_input))

    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)
    response = llm.invoke(
        f"Answer the question based **only** on the provided context. "
        f"If the answer is not in the context, say 'I don't know'.\n\n"
        f"**Context:**\n{context}\n\n"
        f"**Chat History:**\n{chat_history}\n\n"
        f"**User's Question:** {user_input}"
    )

    chat_history.append(AIMessage(content=response.content))
    return response.content

# API Endpoint
@app.post("/api/chat")
async def chat_endpoint(chat_request: ChatRequest):
    response = get_response(chat_request.message)
    return {"answer": response}
