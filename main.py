import streamlit as st
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate

# Set page configuration
st.set_page_config(page_title="HUMAN AI Chatbot", page_icon="🤖", layout="centered")

# Custom CSS for chat bubbles
st.markdown("""
    <style>
        .user-msg { 
            background-color: #DCF8C6; 
            padding: 10px 15px; 
            border-radius: 12px; 
            margin-bottom: 10px; 
            text-align: left; 
            color: black;
        }
        .bot-msg { 
            background-color: #F1F0F0; 
            padding: 10px 15px; 
            border-radius: 12px; 
            margin-bottom: 10px; 
            text-align: left; 
            color: black;
        }
        .chat-container {
            max-height: 500px;
            overflow-y: auto;
            padding-right: 10px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🤖 HUMAN AI Chatbot")
st.write("Ask anything and let HUMAN assist you intelligently!")

# Template for prompt
template = """
Answer the question below based on the provided context.

Context:
{context}

Question:
{question}

Answer:
"""

# Initialize model and prompt
model = OllamaLLM(model="llama3")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# Initialize session state for conversation
if "context" not in st.session_state:
    st.session_state.context = ""
if "history" not in st.session_state:
    st.session_state.history = []

# Input box
user_input = st.chat_input("Type your message...")

# On message input
if user_input:
    result = chain.invoke({
        "context": st.session_state.context,
        "question": user_input
    })

    # Update context and history
    st.session_state.context += f"\nUser: {user_input}\nBot: {result}"
    st.session_state.history.append(("You", user_input))
    st.session_state.history.append(("HUMAN", result))

# Show conversation history in styled bubbles
with st.container():
    for sender, msg in st.session_state.history:
        if sender == "You":
            st.markdown(f"<div class='user-msg'><b>{sender}:</b> {msg}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='bot-msg'><b>{sender}:</b> {msg}</div>", unsafe_allow_html=True)

# Optional: Sidebar history or clear button
with st.sidebar:
    st.header("🧠 Chat History")
    if st.button("🗑️ Clear Chat"):
        st.session_state.history = []
        st.session_state.context = ""
        st.rerun()
