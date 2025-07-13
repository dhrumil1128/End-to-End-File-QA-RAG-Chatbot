
# app.py

# Import the core LangChain library for building LLM applications
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.callbacks.base import BaseCallbackHandler

# Import components for integrating with Google's Generative AI (Gemini) models
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Import LangChain Community integrations
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Chroma

# Import LangChain text splitting utility
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import standard Python libraries
from operator import itemgetter
import streamlit as st
import tempfile
import os
import pandas as pd
import yaml # Make sure this import is present

# --- Set page config at the very top ---
st.set_page_config(page_title="File QA Chatbot", page_icon="🤖")


# Define the path to your API key file
API_KEY_FILE = "gemini_api_key.yml" # Make sure this matches the file name you upload




# Try to get the key from environment variables (Colab Secrets) first
env_api_key = os.getenv("GOOGLE_API_KEY")

if env_api_key:
    print(f"DEBUG: GOOGLE_API_KEY IS found in environment variables by app.py! Starts with: {env_api_key[:5]}*****")
  
else:
    print("DEBUG: GOOGLE_API_KEY NOT found in environment variables. Checking file...")
    # If not in env, try to load from the YAML file
    if os.path.exists(API_KEY_FILE):
        try:
            with open(API_KEY_FILE, 'r') as f:
                config = yaml.safe_load(f)
                file_api_key = config.get("GOOGLE_API_KEY") # Get key from the YAML file
                if file_api_key:
                    os.environ["GOOGLE_API_KEY"] = file_api_key # Set it as an environment variable for LangChain
                    print(f"DEBUG: GOOGLE_API_KEY IS found in '{API_KEY_FILE}' by app.py! Starts with: {file_api_key[:5]}*****")

                else:
                    print(f"DEBUG: GOOGLE_API_KEY not found inside '{API_KEY_FILE}'.")
                    st.error(f"Google API Key not found inside {API_KEY_FILE}. Check its content.")
                    st.stop()
        except Exception as e:
            print(f"DEBUG: Error loading API key from file: {e}")
            st.error(f"Error loading API key from {API_KEY_FILE}: {e}")
            st.stop()
    else:
        print(f"DEBUG: '{API_KEY_FILE}' not found in the current directory.")
        st.error(f"Google API Key not found. Please ensure '{API_KEY_FILE}' is uploaded or set in Colab Secrets.")
        st.info("Upload 'gemini_api_key.yml' with your key, or set 'GOOGLE_API_KEY' in Colab Secrets.")
        st.stop()


# UI Part :
st.title("Welcome to File QA RAG Chatbot 🤖")

@st.cache_resource(ttl="1h")
def configure_retriever(uploaded_files):
    # Read documents
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        loader = PyMuPDFLoader(temp_filepath)
        docs.extend(loader.load())

    # Split into documents chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    doc_chunks = text_splitter.split_documents(docs)

    # Create document embeddings and Store in Vector DB
    # This line will now definitely find GOOGLE_API_KEY if the loading logic above was successful
    embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001") # Corrected
    vectordb = Chroma.from_documents(doc_chunks, embeddings_model)
    

    # Define retriever object
    retriever = vectordb.as_retriever()
    return retriever

# Manages live updates to a Streamlit app's display by appending new text tokens
# to an existing text stream and rendering the updated text in Markdown
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

# Creates UI elements to accept PDF uploads
uploaded_files = st.sidebar.file_uploader(
    label="Upload PDF files", type=["pdf"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("Please upload PDF documents to continue.")
    st.stop()

# Create retriever object based on uploaded PDFs
retriever = configure_retriever(uploaded_files)

# Load a connection to Gemini LLM
gemini = ChatGoogleGenerativeAI(model='gemini-1.5-flash',
                                 temperature=0.2,
                                 disable_streaming=False) # Corrected

# Create a prompt template for QA RAG System
qa_template = """
                    Use only the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know,
                    don't try to make up an answer. Keep the answer as concise as possible.

                    {context}

                    Question: {question}
                    """
qa_prompt = ChatPromptTemplate.from_template(qa_template)

# this function formats retrieved document before sending to LLM
def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs]) # Corrected

#Create a QA RAG System Chain (pipeline)
qa_rag_chain = (
    {
    "context": itemgetter("question") # based on the user question get context docs
            |
    retriever
            |
    format_docs,

    "question": itemgetter("question") # user question
    }
            |
    qa_prompt # prompt with above user question and context
            |
    gemini  # above prompt is sent to the LLM for response
    )

# store conversation history in Streamlit session state
streamlit_msg_history = StreamlitChatMessageHistory(key="langchain_message")

# show the first message when app starts
if len(streamlit_msg_history.messages) == 0:
    streamlit_msg_history.add_ai_message("Please ask your question ? ")

# Render current message from StreamlitChatMessageHistory
for msg in streamlit_msg_history.messages:
    st.chat_message(msg.type).write(msg.content)

# Callback handler which does some post-processing on the LLM response
# Used to post the top 3 document sources used by the LLM in RAG response
class PostMessageHandler(BaseCallbackHandler):
    def __init__(self, msg: st.write):
        BaseCallbackHandler.__init__(self)
        self.msg = msg
        self.sources = []

    def on_retriever_end(self, documents, *, run_id, parent_run_id, **kwargs):
        source_ids = []
        for d in documents: # retrieved documents from retriever based on user query
            metadata = {
                "source": d.metadata["source"],
                "page": d.metadata["page"],
                "content": d.page_content[:200]
            }
            idx = (metadata["source"], metadata["page"])
            if idx not in source_ids: # store unique source documents
                source_ids.append(idx)
                self.sources.append(metadata)


    def on_llm_end(self, response, *, run_id, parent_run_id, **kwargs):
        if len(self.sources):
            st.markdown("__Sources:__ " + "\n")
            st.dataframe(data=pd.DataFrame(self.sources[:3]),
                        width=1000) # Top 3 sources


# if user inputs a new prompt, display it and show the response
if user_prompt := st.chat_input():
    st.chat_message("human").write(user_prompt)
    # This is where response from the LLM is shown
    with st.chat_message("ai"):
        # initializing an empty data stream
        stream_handler = StreamHandler(st.empty())
        # UI element to write RAG sources after LLM response
        sources_container = st.write("")
        pm_handler = PostMessageHandler(sources_container)
        config = {"callbacks": [stream_handler, pm_handler]}
        # Get LLM response
        response = qa_rag_chain.invoke({"question": user_prompt},
                                        config=config) # Changed to config=config

        # --- ADDED DEBUGGING LINES HERE ---
        # First, print the entire response object to Colab logs
        print(f"DEBUG: Full response object from chain.invoke: {response}")

        # Check if the streamed output (from StreamHandler) is empty.
        # If the streamed output from StreamHandler is empty, display the final response content directly.
        # This acts as a fallback to ensure the answer is shown even if streaming fails.
        if not stream_handler.text: # Check if StreamHandler collected any text
            if hasattr(response, 'content'): # If the response is a LangChain message object
                st.markdown(response.content)
                print("DEBUG: Displayed response.content as fallback.")
            elif isinstance(response, str): # If the response is just a string
                st.markdown(response)
                print("DEBUG: Displayed response as fallback string.")
            else:
                print(f"DEBUG: Response object type not recognized for direct display: {type(response)}")
        else:
            print("DEBUG: StreamHandler successfully collected text. Fallback not needed.")
        # --- END DEBUGGING ADDITION ---
