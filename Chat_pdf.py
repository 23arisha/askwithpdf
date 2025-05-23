import warnings
import logging
import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
import tempfile
from duckduckgo_search import DDGS
import time

# Disable warnings and info logs
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# Streamlit Title
st.title('📚 AskMyPDF')
st.markdown(
    "Upload a PDF document and ask questions about its content. "
    "If the answer is not found or the question is out of context, it can search the web (if enabled)."
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "pdf_uploaded" not in st.session_state:
    st.session_state.pdf_uploaded = False
if "web_search_enabled" not in st.session_state:
    st.session_state.web_search_enabled = False

# Display previous chat history
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

# Function for web search using DuckDuckGo
def web_search(query, max_results=4, retries=3, delay=2):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0"
    }
    for attempt in range(retries):
        try:
            with DDGS(headers=headers) as ddgs:
                results = ddgs.text(query, max_results=max_results)
                return "\n".join(
                    [f"- Title: {r['title']}\n  Summary: {r['body']}\n  URL: {r['href']}" for r in results]
                )
        except Exception as e:
            time.sleep(delay)
            if attempt == retries - 1:
                return f"Web search failed after {retries} attempts: {e}"

# Function to create vectorstore from PDF
def get_vectorstore(uploaded_file=None):
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name
        pdf_loader = PyPDFLoader(temp_file_path)
    else:
        pdf_loader = PyPDFLoader("./cnn_doc.pdf")  # fallback
    vectordb = VectorstoreIndexCreator(
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100),
        embedding=HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            cache_folder="./hf_model_cache"
        ),
    ).from_loaders([pdf_loader])
    return vectordb.vectorstore

# --- Phase 1: Upload interface ---
if not st.session_state.pdf_uploaded:
    uploaded_file = st.file_uploader("📎 Upload a PDF", type="pdf")
    if uploaded_file:
        st.session_state.vectorstore = get_vectorstore(uploaded_file)
        st.session_state.pdf_uploaded = True
        st.rerun()

# --- Phase 2: Chat interface ---
if st.session_state.pdf_uploaded:
    # Persistent checkbox
    st.session_state.web_search_enabled = st.checkbox(
        "Enable web search if answer is not found in the PDF",
        value=st.session_state.web_search_enabled
    )

    prompt = st.chat_input("Ask a question about the PDF document:")
    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        # Set up model
        model = "llama3-8b-8192"
        chat_model = ChatGroq(
            groq_api_key=st.secrets["GROQ_API_KEY"],
            model_name=model
        )

        try:
            # === Relevance Classifier ===
            top_doc = st.session_state.vectorstore.similarity_search(prompt, k=1)[0]
            relevance_prompt = f"""
You are an assistant helping to determine if a user's question can be answered using a document.

Document excerpt:
\"\"\"{top_doc.page_content[:1000]}\"\"\"

User question:
"{prompt}"

Based on the document, can this question be answered directly from it? Answer only "Yes" or "No".
"""
            relevance_answer = chat_model.invoke(relevance_prompt).content.strip().lower()

            if "no" in relevance_answer:
                st.chat_message("assistant").markdown("❌ This question appears unrelated to the PDF content.")

                if st.session_state.web_search_enabled:
                    st.chat_message("assistant").markdown("🔎 Searching the web for a relevant answer...")
                    web_info = web_search(prompt)

                    full_prompt = f"""
The user asked: '{prompt}'.

Here are the top relevant search results:
{web_info}

Based on these results, provide a clear and accurate answer.
If no relevant answer is found, say so.
"""
                    llm_response = chat_model.invoke(full_prompt)
                    response = llm_response.content.strip() if hasattr(llm_response, "content") else str(llm_response).strip()
                    response = response.replace("\\n", " ").replace("\n", " ").replace("  ", " ")
                else:
                    response = "🌐 Web search is disabled. Please refine your question or enable search above."

            else:
                # === Run normal RetrievalQA ===
                chain = RetrievalQA.from_chain_type(
                    llm=chat_model,
                    retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3}),
                    return_source_documents=True,
                    chain_type="stuff"
                )

                result = chain({"query": prompt})
                response = result['result'].strip()

            # Display response
            st.chat_message("assistant").markdown(response)
            st.session_state.messages.append({'role': 'assistant', 'content': response})

        except Exception as e:
            st.error(f"An error occurred: {e}")
