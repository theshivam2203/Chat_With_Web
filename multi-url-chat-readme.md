# Multi-URL Website Chat Assistant

A Streamlit-based application that allows users to chat with multiple websites simultaneously using AI. The application combines web scraping, vector embeddings, and the Gemini API to provide intelligent responses based on website content.

## Features

- Process and chat with multiple websites simultaneously
- Real-time web content extraction and processing
- AI-powered responses using Google's Gemini API
- Vector-based search for relevant content retrieval
- User-friendly chat interface
- Source attribution in responses
- Persistent session management

## Prerequisites

- Python 3.8 or higher
- Streamlit
- Google Gemini API key
- Internet connection

## Required Packages

```bash
streamlit
google-generativeai
langchain
langchain-community
langchain-huggingface
faiss-cpu
beautifulsoup4
requests
nest-asyncio
sentence-transformers
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd multi-url-chat-assistant
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Set up your environment:
   - Create a `.streamlit/secrets.toml` file
   - Add your Gemini API key:
     ```toml
     GEMINI_API_KEY = "your-api-key-here"
     ```

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Using the application:
   - Enter website URLs one at a time in the URL input field
   - Click "Add URL" to process each website
   - Wait for confirmation that the website has been processed
   - Use the chat interface to ask questions about the websites
   - View responses that include information from all processed websites

## Application Structure

- `app.py`: Main application file containing all the code
- `vectorstore_web/`: Directory for storing vector embeddings
- `db_faiss/`: FAISS database storage directory

## Key Components

### MultiURLVectorDatabase Class
Handles the processing and storage of multiple website contents:
- URL validation and processing
- Content extraction and cleaning
- Vector embeddings creation and storage
- Response generation using Gemini API

### Main Functions

- `is_valid_url()`: Validates URL format
- `get_webpage_content()`: Extracts and cleans webpage content
- `process_url()`: Processes individual URLs
- `process_urls()`: Handles multiple URL processing
- `generate_response()`: Creates AI responses based on stored content

## Error Handling

The application includes comprehensive error handling for:
- Invalid URLs
- Failed web requests
- Content processing errors
- API communication issues
- Database operations

## Limitations

- Requires stable internet connection
- May not work with JavaScript-heavy websites
- Processing time increases with number of websites
- Limited by Gemini API rate limits
- Text-only content processing (no images or dynamic content)

## Troubleshooting

1. If the application fails to start:
   - Check if all dependencies are installed
   - Verify Gemini API key is correctly set
   - Ensure Python version compatibility

2. If website processing fails:
   - Check URL validity
   - Verify website accessibility
   - Check internet connection
   - Review logs for specific errors

3. If chat responses are slow:
   - Reduce number of processed websites
   - Check internet connection speed
   - Verify API key rate limits

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

[Add your chosen license here]

## Acknowledgments

- Google Gemini API
- Streamlit
- LangChain
- FAISS
- BeautifulSoup4
- All other open-source contributors

## Support

For support, please:
1. Check the troubleshooting section
2. Review existing issues
3. Create a new issue with detailed error description
4. Contact the maintainers

---

**Note:** Remember to replace `<repository-url>` with your actual repository URL and add appropriate license information before deploying.
