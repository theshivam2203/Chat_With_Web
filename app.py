import streamlit as st
import google.generativeai as genai
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from crawl4ai import WebCrawler
from urllib.parse import urlparse
import os
import shutil
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DB_FAISS_PATH = 'vectorstore_web/db_faiss'

# Configure Gemini API using Streamlit secrets
if 'GEMINI_API_KEY' in st.secrets:
    genai.configure(api_key=st.secrets['GEMINI_API_KEY'])
else:
    st.error("GEMINI_API_KEY not found in Streamlit secrets.")
    st.stop()

def is_valid_url(url):
    try:
        url = url.replace('www.', '', 1)
        if not urlparse(url).scheme:
            url = 'https://' + url
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception as e:
        logger.error(f"URL validation failed: {e}")
        return False

class VectorDatabase:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            model_kwargs={'device': 'cpu'}  # Changed to CPU for cloud compatibility
        )
        self.db = None

    def create_or_update_database(self, url):
        try:
            url = url.replace('www.', '', 1)
            if not url.startswith("http://") and not url.startswith("https://"):
                url = "https://" + url

            logger.info(f"Fetching documents from URL: {url}")
            crawler = WebCrawler()
            crawler.warmup()
            result = crawler.run(url)

            if not result or not result.markdown:
                logger.warning(f"No content fetched from URL: {url}")
                return None

            documents = [Document(page_content=result.markdown, metadata={"source": url})]
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            texts = text_splitter.split_documents(documents)

            if not texts:
                logger.error(f"No texts extracted from documents: {url}")
                return None

            if os.path.exists(DB_FAISS_PATH):
                shutil.rmtree(DB_FAISS_PATH)

            self.db = FAISS.from_documents(texts, self.embeddings)
            self.db.save_local(DB_FAISS_PATH)
            return True

        except Exception as exc:
            logger.error(f"Error creating or updating database for URL {url}: {exc}", exc_info=True)
            return None

    def generate_response(self, query):
        try:
            if not self.db:
                return "Please provide a valid URL first."
            
            retrieved_docs = self.db.similarity_search(query, k=3)
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])

            prompt = f"Context:\n{context}\n\nUser: {query}\n\nAI:"
            
            model = genai.GenerativeModel("gemini-pro")
            response = model.generate_content(prompt)

            return response.text if response else "No response generated"
        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True)
            return f"Error: {str(e)}"

def main():
    st.set_page_config(page_title="Website Chat Assistant", page_icon="💬")
    
    st.title("💬 Website Chat Assistant")
    st.write("Chat with any website using AI!")

    # Initialize session state
    if 'vector_db' not in st.session_state:
        st.session_state.vector_db = VectorDatabase()
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'current_url' not in st.session_state:
        st.session_state.current_url = ""

    # URL input
    url = st.text_input("Enter website URL:", key="url_input")

    if url and url != st.session_state.current_url:
        if not is_valid_url(url):
            st.error("Please enter a valid URL")
        else:
            with st.spinner("Processing website content..."):
                success = st.session_state.vector_db.create_or_update_database(url)
                if success:
                    st.session_state.current_url = url
                    st.session_state.messages = []  # Clear chat history for new URL
                    st.success("Website processed successfully! You can now start chatting.")
                else:
                    st.error("Failed to process the website. Please try another URL.")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if query := st.chat_input("Ask something about the website..."):
        if not st.session_state.current_url:
            st.error("Please enter a valid URL first")
            return

        # Add user message
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.vector_db.generate_response(query)
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
