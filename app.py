import streamlit as st
import google.generativeai as genai
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from urllib.parse import urlparse
import os
import shutil
import logging
import requests
from bs4 import BeautifulSoup
import asyncio
import nest_asyncio

# Apply nest_asyncio to handle async operations in Streamlit
nest_asyncio.apply()

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
    """
    Validate if the given string is a proper URL.
    Returns True if valid, False otherwise.
    """
    try:
        url = url.replace('www.', '', 1)
        if not urlparse(url).scheme:
            url = 'https://' + url
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception as e:
        logger.error(f"URL validation failed: {e}")
        return False

def get_webpage_content(url):
    """
    Fetch and extract clean text content from a webpage.
    Returns the cleaned text content or None if failed.
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
            
        # Get text content
        text = soup.get_text(separator='\n', strip=True)
        
        # Clean up text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text
    except Exception as e:
        logger.error(f"Error fetching webpage content: {e}")
        return None

class MultiURLVectorDatabase:
    """
    Enhanced vector database class that handles multiple URLs and their content.
    """
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            model_kwargs={'device': 'cpu'}
        )
        self.db = None
        self.processed_urls = set()  # Keep track of processed URLs

    def process_url(self, url):
        """
        Process a single URL and add its content to the vector database.
        Returns True if successful, False otherwise.
        """
        try:
            url = url.replace('www.', '', 1)
            if not url.startswith("http://") and not url.startswith("https://"):
                url = "https://" + url

            logger.info(f"Fetching documents from URL: {url}")
            content = get_webpage_content(url)

            if not content:
                logger.warning(f"No content fetched from URL: {url}")
                return False

            documents = [Document(page_content=content, metadata={"source": url})]
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            texts = text_splitter.split_documents(documents)

            if not texts:
                logger.error(f"No texts extracted from documents: {url}")
                return False

            # Add to existing database or create new one
            if self.db is None:
                self.db = FAISS.from_documents(texts, self.embeddings)
            else:
                self.db.add_documents(texts)

            self.processed_urls.add(url)
            return True

        except Exception as exc:
            logger.error(f"Error processing URL {url}: {exc}", exc_info=True)
            return False

    def process_urls(self, urls):
        """
        Process multiple URLs and update the vector database.
        Returns list of successfully processed URLs.
        """
        successful_urls = []
        for url in urls:
            if url not in self.processed_urls:  # Only process new URLs
                if self.process_url(url):
                    successful_urls.append(url)
        
        # Save the updated database
        if successful_urls and self.db:
            if os.path.exists(DB_FAISS_PATH):
                shutil.rmtree(DB_FAISS_PATH)
            self.db.save_local(DB_FAISS_PATH)
        
        return successful_urls

    def generate_response(self, query):
        """
        Generate a response based on the query using content from all processed URLs.
        """
        try:
            if not self.db:
                return "Please provide at least one valid URL first."
            
            retrieved_docs = self.db.similarity_search(query, k=3)
            context = "\n\n".join([
                f"From {doc.metadata['source']}:\n{doc.page_content}" 
                for doc in retrieved_docs
            ])

            prompt = f"Context from multiple sources:\n{context}\n\nUser: {query}\n\nAI:"
            
            model = genai.GenerativeModel("gemini-pro")
            response = model.generate_content(prompt)

            return response.text if response else "No response generated"
        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True)
            return f"Error: {str(e)}"

def main():
    st.set_page_config(page_title="Multi-Website Chat Assistant", page_icon="💬")
    
    st.title("💬 Multi-Website Chat Assistant")
    st.write("Chat with multiple websites using AI!")

    # Initialize session state
    if 'vector_db' not in st.session_state:
        st.session_state.vector_db = MultiURLVectorDatabase()
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'urls' not in st.session_state:
        st.session_state.urls = []

    # URL input section
    st.subheader("Add Websites")
    url = st.text_input("Enter website URL:")
    if st.button("Add URL"):
        if not url:
            st.warning("Please enter a URL")
        elif not is_valid_url(url):
            st.error("Please enter a valid URL")
        elif url in st.session_state.urls:
            st.warning("This URL has already been added")
        else:
            with st.spinner("Processing website content..."):
                if st.session_state.vector_db.process_url(url):
                    st.session_state.urls.append(url)
                    st.success(f"Successfully added: {url}")
                else:
                    st.error(f"Failed to process: {url}")

    # Display processed URLs
    if st.session_state.urls:
        st.subheader("Processed Websites")
        for processed_url in st.session_state.urls:
            st.write(f"- {processed_url}")

    # Chat interface
    st.subheader("Chat")
    if not st.session_state.urls:
        st.info("Please add at least one website to start chatting")
    else:
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # Chat input
        if query := st.chat_input("Ask something about the websites..."):
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
