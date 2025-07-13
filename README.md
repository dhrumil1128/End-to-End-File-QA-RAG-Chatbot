# End-to-End File QA RAG Chatbot

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32.2-FF4B4B?style=flat-square&logo=streamlit)
![LangChain](https://img.shields.io/badge/LangChain-0.1.12-green?style=flat-square&logo=chainlink)
![Google Gemini](https://img.shields.io/badge/Google_Gemini-1.5_Flash-orange?style=flat-square&logo=google)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.4.24-purple?style=flat-square&logo=chroma)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

---

## 🚀 Project Overview

This project implements an **End-to-End File Question-Answering (QA) Chatbot** powered by Retrieval Augmented Generation (RAG). It allows users to upload multiple PDF documents, which are then processed and indexed to create a searchable knowledge base. The chatbot leverages Google's Gemini Large Language Models (LLMs) via LangChain to answer user queries based *only* on the content of the uploaded documents, providing accurate, context-aware responses and citing the source documents.

The application is built with Streamlit for an intuitive user interface, making it easy to interact with the RAG system.

## ✨ Features

* **PDF Document Upload:** Seamlessly upload one or multiple PDF files via the Streamlit interface.
* **Document Indexing:** Automatically processes uploaded PDFs by splitting them into manageable chunks and generating vector embeddings.
* **Vector Database Integration:** Utilizes ChromaDB to efficiently store and retrieve document embeddings.
* **Retrieval Augmented Generation (RAG):** Enhances LLM responses by retrieving relevant document snippets based on user queries, ensuring answers are grounded in the provided context.
* **Google Gemini Integration:** Leverages `gemini-1.5-flash` for powerful and concise text generation.
* **Real-time Response Streaming:** Provides a dynamic, token-by-token display of LLM responses for a better user experience.
* **Source Citation:** Clearly displays the top 3 relevant source documents (including page numbers) used to formulate the answer.
* **Chat History:** Maintains conversation history within the Streamlit session.

## 🛠️ Technologies Used

* **Python 3.11+**: The core programming language.
* **Streamlit**: For building interactive web applications and the user interface.
* **LangChain**: A framework for developing applications powered by LLMs, used for:
    * `langchain-core`: Core components for LLM applications.
    * `langchain-google-genai`: Integration with Google's Generative AI models (Gemini).
    * `langchain-community`: Community integrations, including `StreamlitChatMessageHistory` and `PyMuPDFLoader`.
    * `langchain-text-splitters`: For efficient document chunking.
* **Google Generative AI (Gemini API)**:
    * **`gemini-1.5-flash`**: The LLM used for generating responses.
    * **`models/embedding-001`**: The embedding model used for creating vector representations of text.
* **ChromaDB**: An open-source embedding database for storing and querying vector embeddings.
* **PyMuPDF (fitz)**: For loading and parsing PDF documents.
* **pandas**: For structured data handling, especially for displaying sources.
* **PyYAML**: For securely loading API keys from a YAML file (primarily for local development/Colab).
* **`tempfile`**: For handling temporary file storage during PDF processing.
* **`os`**: For operating system interactions, including environment variables.
* **`operator` (itemgetter)**: For efficient data extraction in LangChain chains.

## ⚙️ Architecture & How It Works

The chatbot operates on a RAG architecture, which involves the following steps:

1.  **Document Ingestion:**
    * User uploads PDF files via the Streamlit sidebar.
    * `PyMuPDFLoader` reads the content of these PDFs.
    * `RecursiveCharacterTextSplitter` breaks down the large documents into smaller, manageable `doc_chunks`.
2.  **Embedding & Vector Storage:**
    * `GoogleGenerativeAIEmbeddings` generates numerical vector representations (embeddings) for each `doc_chunk`.
    * These embeddings, along with their original text content, are stored in a `Chroma` vector database.
3.  **Retrieval:**
    * When a user asks a question, the `retriever` (configured from the `Chroma` vector store) searches for and retrieves the most semantically relevant `doc_chunks` from the vector database.
4.  **Generation:**
    * The retrieved `doc_chunks` (context) are combined with the user's `question` into a `ChatPromptTemplate`.
    * This prompt is then sent to the `ChatGoogleGenerativeAI` (`gemini-1.5-flash`) LLM.
    * The LLM generates a concise answer based *only* on the provided context.
5.  **Streaming & Display:**
    * The LLM's response is streamed back token by token using a `StreamHandler`.
    * A `PostMessageHandler` extracts metadata from the retrieved documents to display the top 3 sources in a `pandas` DataFrame below the answer.

## 🚀 Demo

This application is designed for deployment as a web service for continuous availability.

**Live Demo:** [**https://your-render-app-url.onrender.com**](https://your-render-app-url.onrender.com)
*(**Important:** Please replace `https://your-render-app-url.onrender.com` with the actual URL you get after deploying your application on Render.)*

---

## 💻 Local Setup & Installation

To run this application locally on your machine:

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```
    *(Remember to replace `your-username/your-repo-name` with your actual GitHub repository details.)*

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    # .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Ensure `requirements.txt` contains all necessary packages like `streamlit`, `langchain-google-genai`, `PyMuPDF`, `chromadb`, `pyyaml`, `pandas`, etc. You can generate it using `pip freeze > requirements.txt` in your working environment.)*

4.  **Set Up Your Google Gemini API Key:**
    **IMPORTANT: DO NOT commit your API key directly to your GitHub repository.**

    * **Option A: Environment Variable (Recommended for Local Dev)**
        Set your Google API Key as an environment variable named `GOOGLE_API_KEY`.
        * **Linux/macOS:**
            ```bash
            export GOOGLE_API_KEY="YOUR_ACTUAL_GOOGLE_GEMINI_API_KEY"
            ```
            (For persistence across sessions, add this line to your `~/.bashrc` or `~/.zshrc` file and then run `source ~/.bashrc` or `source ~/.zshrc`).
        * **Windows (Command Prompt):**
            ```cmd
            set GOOGLE_API_KEY="YOUR_ACTUAL_GOOGLE_GEMINI_API_KEY"
            ```
            (This is temporary for the current session. For persistence, you'll need to set it in System Environment Variables via the Control Panel).

    * **Option B: `gemini_api_key.yml` file (For Colab or specific local setups, ensure `.gitignore`!)**
        Create a file named `gemini_api_key.yml` in the root directory of your project (same level as `app.py`).
        Add your API key to it in the following format:
        ```yaml
        GOOGLE_API_KEY: YOUR_ACTUAL_GOOGLE_GEMINI_API_KEY
        ```
        **Crucially, add `gemini_api_key.yml` to your `.gitignore` file to prevent accidental commits of your key.**

5.  **Run the Streamlit Application:**
    ```bash
    streamlit run app.py
    ```
    This will open the application in your web browser, typically at `http://localhost:8501`.

---

## ☁️ Deployment on Render

This application is ideal for deployment on cloud platforms like Render for a permanent web presence.

1.  **Prepare your GitHub Repository:**
    * Ensure your `app.py` and `requirements.txt` files are in the root of your repository.
    * **Crucially, double-check that `gemini_api_key.yml` (or any file containing your raw API key) is NOT committed to GitHub.** If it is, remove it from version control and add it to `.gitignore`.

2.  **Create a `Procfile` (Optional but Recommended):**
    * In the root of your repository, create a file named `Procfile` (no file extension).
    * Add the following line to instruct Render on how to start your Streamlit app:
        ```
        web: streamlit run app.py --server.port $PORT --server.enableCORS false --server.enableXsrfProtection false
        ```
---

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improvements, bug reports, or would like to contribute code, please feel free to open an issue or submit a pull request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.




