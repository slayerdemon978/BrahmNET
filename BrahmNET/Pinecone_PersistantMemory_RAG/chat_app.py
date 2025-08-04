import os
import time
import streamlit as st
from langchain_cohere import CohereEmbeddings
from pinecone import Pinecone
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="Deposition Station Chat", layout="wide")
st.title("Deposition Station Chat")

st.header("Usage Instructions")
st.info("""
**Welcome to Deposition Station Chat**

Chat with your **pre-embedded Pinecone index** using an LLM-powered RAG pipeline.

**Tech stack:**  
- **LLM:** Groq `meta-llama/llama-4-scout-17b-16e-instruct`  
- **Embeddings:** Cohere `embed-english-light-v3.0`  
- **Vector Store:** Pinecone

**Required API Keys:**  
- [Groq API Key](https://groq.com/)  
- [Cohere API Key](https://cohere.com/)  
- [Pinecone API Key](https://www.pinecone.io/)

**Rules:**  
- Provide **all three keys** below  
- Never share your keys  
- Your Pinecone index must already exist and contain embeddings
""")

with st.sidebar:
    st.header("Required API Keys")
    groq_key = st.text_input("Groq API Key", type="password")
    cohere_key = st.text_input("Cohere API Key", type="password")
    pinecone_key = st.text_input("Pinecone API Key", type="password")
    keys_submit = st.button("Submit API Keys")

if "keys_valid" not in st.session_state:
    st.session_state.keys_valid = False

if keys_submit:
    if groq_key and cohere_key and pinecone_key:
        os.environ["GROQ_API_KEY"] = groq_key
        os.environ["COHERE_API_KEY"] = cohere_key
        os.environ["PINECONE_API_KEY"] = pinecone_key
        st.session_state.keys_valid = True
        st.success("API keys saved. You can now use the chat below.")
    else:
        st.error("Please enter **all three** API keys and click Submit.")

if st.session_state.keys_valid:
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    embeddings = CohereEmbeddings(model="embed-english-light-v3.0")
    model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.1)
    parser = StrOutputParser()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    index_name = st.text_input("Enter Pinecone Index Name")

    if index_name:
        indexes = [i.name for i in pc.list_indexes()]
        if index_name in indexes:
            index = pc.Index(index_name)
            retriever = PineconeVectorStore(index=index, embedding=embeddings, text_key="text").as_retriever()
            retriever.search_kwargs["k"] = 5

            user_input = st.chat_input("Ask your question...")

            for q, a in st.session_state.chat_history:
                with st.chat_message("user"):
                    st.write(q)
                with st.chat_message("assistant"):
                    st.write(a)

            if user_input:
                with st.chat_message("user"):
                    st.write(user_input)

                with st.spinner("Thinking..."):
                    docs = retriever.get_relevant_documents(user_input)
                    context = "\n\n".join([doc.page_content for doc in docs])
                    prompt = f"""Use this context to answer crisply.

Context:
{context}

Question: {user_input}

Answer:"""
                    response = model.invoke(prompt)
                    answer = parser.invoke(response)

                st.session_state.chat_history.append((user_input, answer))

                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    final_text = ""
                    for chunk in answer:
                        final_text += chunk
                        message_placeholder.markdown(final_text + "▌")
                        time.sleep(0.02)
                    message_placeholder.markdown(final_text)
        else:
            st.warning("Index not found. Please check your Pinecone index name.")
