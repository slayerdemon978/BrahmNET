import os
import re
import tempfile
import streamlit as st
from langchain_community.document_loaders import TextLoader, PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_community.vectorstores import Pinecone as PineconeVectorStore

st.set_page_config(page_title="RishiGPT Embedding Station", layout="wide")
st.title("RishiGPT Persistent File Deposition Station")

st.header("Usage Rules & Setup")
st.info("""
**RishiGPT Embedding Station**

Use this tool to embed your files or website content into a **Pinecone vector index** with Cohere embeddings.
NOTE: Only 4 indexes allowed on a free plan of Pinecone.

**Requirements:**
- [Cohere API Key](https://cohere.com/) for embeddings (`embed-english-light-v3.0`)
- [Pinecone API Key](https://www.pinecone.io/) to create and store your vector index

**All API keys must be provided below.**
""")

with st.sidebar:
    st.header("Required API Keys")
    cohere_key = st.text_input("Cohere API Key", type="password")
    pinecone_key = st.text_input("Pinecone API Key", type="password")
    keys_submit = st.button("Submit API Keys")

if "keys_valid" not in st.session_state:
    st.session_state.keys_valid = False

if keys_submit:
    if cohere_key and pinecone_key:
        os.environ["COHERE_API_KEY"] = cohere_key
        os.environ["PINECONE_API_KEY"] = pinecone_key
        st.session_state.keys_valid = True
        st.success("API keys saved. Continue below to embed your files or websites.")
    else:
        st.error("Please enter **both** Cohere and Pinecone API keys before submitting.")

if st.session_state.keys_valid:
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    embeddings = CohereEmbeddings(model="embed-english-light-v3.0")

    index_name = st.text_input("Enter Pinecone Index Name (lowercase, hyphens ok)")

    source_type = st.selectbox("Select file source", ["Choose...", "Text", "PDF", "Website"])

    uploaded_file = None
    url_input = None

    if source_type == "Text":
        uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])
    elif source_type == "PDF":
        uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    elif source_type == "Website":
        url_input = st.text_input("Enter a valid HTTPS URL")

    if index_name and source_type != "Choose...":
        run_embed = st.button("Create Index & Embed Now")

        if run_embed:
            if not re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", index_name):
                st.warning("Invalid index name. Use lowercase letters, numbers, and single hyphens only.")
            elif index_name in pc.list_indexes():
                st.warning("Index name already exists. Choose a different name.")
            elif uploaded_file or url_input:
                pc.create_index(
                    index_name,
                    dimension=384,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
                st.success(f"Index '{index_name}' created.")

                index = pc.Index(index_name)
                db = PineconeVectorStore(index=index, embedding=embeddings, text_key="text")

                if uploaded_file:
                    suffix = ".txt" if source_type == "Text" else ".pdf"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_path = tmp_file.name

                    loader = TextLoader(tmp_path) if source_type == "Text" else PyPDFLoader(tmp_path)
                    docs = loader.load()
                    splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
                    db.add_documents(splits)
                    os.remove(tmp_path)
                    st.success(f"File '{uploaded_file.name}' embedded to index '{index_name}'")

                elif url_input:
                    if not url_input.startswith("https://"):
                        st.warning("URL must start with https://")
                    else:
                        loader = WebBaseLoader(url_input)
                        docs = loader.load()
                        splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
                        db.add_documents(splits)
                        st.success(f"Website content embedded to index '{index_name}'")

            else:
                st.warning("Please upload a file or enter a URL.")
