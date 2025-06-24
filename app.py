import os
import streamlit as st
import requests
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.language_models.llms import LLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# 🔑 Load DeepSeek API key
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    st.error("❌ DeepSeek API key not found in .env file. Please set DEEPSEEK_API_KEY.")
    st.stop()

# 🔧 Define DeepSeek-compatible LLM class
# ✅ Replace with OpenRouter version for free usage
#define custom class so langchain can use deepseek
class DeepSeekLLM(LLM):
    @property
    def _llm_type(self):
        return "deepseek_openrouter"

    def _call(self, prompt, stop=None):
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "deepseek/deepseek-chat-v3-0324:free",  # <- FREE MODEL
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.5
                }
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            raise Exception(f"OpenRouter API Error: {str(e)}")#sends prompt to openrouter


# 🧠 Set LLM instance
llm = DeepSeekLLM()

# 🌐 Load and embed webpage
def load_website(url):
    docs = WebBaseLoader(url).load()
    chunks = RecursiveCharacterTextSplitter().split_documents(docs)#breaks into parts
    #convert text into vector
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma.from_documents(chunks, embeddings)#store vectors in chroma for retreival

# 🔄 Build RAG Chain
def build_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever()

    history_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder("chat_history"),
        ("user", "{input}"),
        ("user", "Based on the above, create a search query."),
    ])
    retriever_chain = create_history_aware_retriever(llm, retriever, history_prompt)#retriver und current quest and previous chat

    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer based on this context:\n\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("user", "{input}"),
    ])
    answer_chain = create_stuff_documents_chain(llm, answer_prompt)

    return create_retrieval_chain(retriever_chain, answer_chain)

# 💬 Get model response
def get_response(user_input):
    rag_chain = build_rag_chain(st.session_state.vectorstore)
    return rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })['answer']

# 🖥️ Streamlit UI
st.set_page_config(page_title="Link-N-Chat 🌐", page_icon="🌐")
st.title("Link-N-Chat 🤖 (Powered by DeepSeek)")

with st.sidebar:
    url = st.text_input("🔗 Enter website URL")

if url:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [AIMessage(content="Hi! Ask me anything about this website.")]
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = load_website(url)

    user_msg = st.chat_input("Ask a question about the site...")
    if user_msg:
        reply = get_response(user_msg)
        st.session_state.chat_history.extend([
            HumanMessage(content=user_msg),
            AIMessage(content=reply)
        ])

    for msg in st.session_state.chat_history:
        with st.chat_message("AI" if isinstance(msg, AIMessage) else "Human"):
            st.write(msg.content)
else:
    st.info("ℹ️ Please enter a valid website URL in the sidebar.")
