import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from streamlit_option_menu import option_menu

load_dotenv()

st.set_page_config(page_title="Chat with Websites", layout="wide") 


st.markdown("""
<style>
@keyframes gradient {
    0% { background-position: 0% 50% }
    50% { background-position: 100% 50% }
    100% { background-position: 0% 50% } 
}

body {
  font-family: 'Arial', sans-serif;
  margin: 0;
  padding: 20px;
  background-color: #f0f0f0;  
}

.title {
    font-size: 60px;
    background: linear-gradient(-45deg,  #2c3e50, #3498db, #e74c3c, #9b59b6); 
    background-size: 400% 400%; 
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent; 
    animation: gradient 5s ease infinite;
}

/* Chat Area */
.chat-container {
  width: 80%;
  margin: 30px auto;
  background-color: #fff;
  padding: 20px;
  border-radius: 10px;
  box-shadow: 0px 2px 10px rgba(0, 0, 0, 0.2);
}

.chatbot-response { 
    font-size: 18px;  
    font-weight: bold;
    color: #3498db;  
    text-shadow: 2px 2px 3px rgba(0, 0, 0, 0.1); 
    padding: 15px;
    background-color: #f8f8f8; 
    border-radius: 8px;
    box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.15); 
    margin-bottom: 15px;
}

.user-message {
  text-align: right; 
  margin-bottom: 15px; 
}
.user-message p {
  background-color: #e74c3c;
  color: #fff;
  padding: 15px;
  border-radius: 8px;
}

.chat-input {
  width: 100%; 
  padding: 20px;
  background-color: #fff;
  border: 1px solid #ddd;
  border-radius: 6px;
}
</style>

<h1 class="title">InfoHound</h1> 
""", unsafe_allow_html=True)

st.markdown("Converse with websites, powered by advanced AI for accuracy and understanding.") # Short description

with st.expander("Technical Overview"):
    st.write("InfoHound prioritizes factual accuracy in chatbot responses through Retrieval-Augmented Generation (RAG). It utilizes OpenAI Embeddings for meaning-based search, ensuring queries are understood in context. Chroma provides an efficient knowledge base, allowing WebSage to quickly locate relevant information.  GPT-4 integration enables natural language generation and enhances the fluidity of responses.")

# app config

# Optional: Simple Navigation menu 
with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",   # Set the Menu Title
        options=["Chat", 'Settings'],  # Your Options
        icons=['chat', 'gear'],        # Optional Icons
        default_index=0                # Default Selected Option
    )

def get_vectorstore_from_url(url):
    # get the text in document form
    loader = WebBaseLoader(url)
    document = loader.load()
    
    # split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)
    
    # create a vectorstore from the chunks
    vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings())

    return vector_store

def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI()
    
    retriever = vector_store.as_retriever()
    
    prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
      ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    
    return retriever_chain
    
def get_conversational_rag_chain(retriever_chain): 
    
    llm = ChatOpenAI()
    
    prompt = ChatPromptTemplate.from_messages([
      ("system", "Answer the user's questions based on the below context:\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm,prompt)
    
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_query
    })
    
    return response['answer']


# sidebar
with st.sidebar:
    # st.header("WebSage: Fact-Focused Chatbot") 
    website_url = st.text_input("Please enter the Website URL in the box below")

if website_url is None or website_url == "":
    st.info("Please enter a website URL")

else:
    # session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am a bot. How can I help you?"),
        ]
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vectorstore_from_url(website_url)    

    # user input
    user_query = st.chat_input("Type your message here...")
    if user_query is not None and user_query != "":
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))
       

    # conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)