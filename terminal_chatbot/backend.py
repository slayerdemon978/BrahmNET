import os
import tempfile
from git import Repo, GitCommandError
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader, TextLoader, GitLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from langchain.agents import AgentType, load_tools, initialize_agent
from langchain_core.output_parsers import StrOutputParser
import dotenv

dotenv.load_dotenv("../BrahmNET/RishiGPT/API_KEYS.env")

class ChatbotBackend:
    def __init__(self):
        self.model = ChatGroq(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=0.1
        )
        self.output_parser = StrOutputParser()
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
            k=10
        )
        self.rag_memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
            k=10
        )
        self.rag_mode = None
        self.use_serp = False
        self.vector_store = None
        self.agent = None

    def set_rag_mode(self, rag_mode):
        self.rag_mode = rag_mode

    def set_use_serp(self, use_serp):
        self.use_serp = use_serp
        if use_serp:
            tools = load_tools(["serpapi"], llm=self.model)
            self.agent = initialize_agent(
                llm=self.model,
                tools=tools,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=False,
                memory=None
            )
        else:
            self.agent = None

    def process_file(self, file_path, file_type):
        if file_type == "pdf":
            loader = PyPDFLoader(file_path)
        elif file_type == "txt":
            loader = TextLoader(file_path)
        else:
            raise ValueError("Unsupported file type")

        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = splitter.split_documents(documents)
        embedd = HuggingFaceEmbeddings()
        self.vector_store = FAISS.from_documents(split_docs, embedding=embedd)

    def process_url(self, url):
        loader = WebBaseLoader(url)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = splitter.split_documents(documents)
        embedd = HuggingFaceEmbeddings()
        self.vector_store = FAISS.from_documents(split_docs, embedding=embedd)

    def process_github_repo(self, repo_url):
        repo_folder = tempfile.mkdtemp(prefix="repo_")
        try:
            repo = Repo.clone_from(repo_url, to_path=repo_folder)
            branch = repo.head.reference
            loader = GitLoader(repo_path=repo_folder, branch=branch)
            documents = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            split_docs = splitter.split_documents(documents)
            embedd = HuggingFaceEmbeddings()
            self.vector_store = FAISS.from_documents(split_docs, embedding=embedd)
        except GitCommandError as e:
            raise ValueError(f"Git clone failed: {e}")

    def chat(self, user_query):
        if self.rag_mode and self.vector_store:
            prompt_template = PromptTemplate.from_template(
                "You are a helpful, talkative AI. Be clear, in-depth and give full working code.\nContext: {context}\nQuestion: {question}"
            )
            retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
            chain = ConversationalRetrievalChain.from_llm(
                llm=self.model,
                retriever=retriever,
                memory=self.rag_memory,
                return_source_documents=True,
                combine_docs_chain_kwargs={"prompt": prompt_template}
            )
            response = chain.invoke({"question": user_query})
            return self.output_parser.invoke(response["answer"])
        elif self.use_serp and self.agent:
            serp_response = self.agent.run(user_query)
            response_text = self.output_parser.invoke(serp_response)
            self.memory.chat_memory.add_user_message(user_query)
            self.memory.chat_memory.add_ai_message(response_text)
            return response_text

from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder='static')
backend = ChatbotBackend()

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    data = request.get_json()
    user_query = data.get('query')
    if not user_query:
        return jsonify({'error': 'Query not provided'}), 400

    response = backend.chat(user_query)
    return jsonify({'response': response})

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    file_type = file.filename.split('.')[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type}") as tmp:
        file.save(tmp.name)
        backend.process_file(tmp.name, file_type)

    return jsonify({'message': 'File processed successfully'})

@app.route('/set_mode', methods=['POST'])
def set_mode():
    data = request.get_json()
    rag_mode = data.get('rag_mode')
    use_serp = data.get('use_serp')

    if rag_mode:
        backend.set_rag_mode(rag_mode)
    if use_serp is not None:
        backend.set_use_serp(use_serp)

    return jsonify({'message': 'Mode set successfully'})

if __name__ == '__main__':
    app.run(debug=True)
        else:
            model_response = self.model.invoke(user_query)
            response_text = self.output_parser.invoke(model_response)
            self.memory.chat_memory.add_user_message(user_query)
            self.memory.chat_memory.add_ai_message(response_text)
            return response_text
