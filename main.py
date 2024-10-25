import shutil
import streamlit as st
import sqlite3
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI  # Update to ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import warnings
import os
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

# Load environment variables from .env file
load_dotenv()

data_directory = os.path.join(os.path.dirname(__file__), "data")


# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Set up SQLite connection
def initialize_database():
    conn = sqlite3.connect("fashionbot.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversation_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_query TEXT NOT NULL,
            bot_response TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn

db_connection = initialize_database()

# Function to log conversation
def log_conversation(user_query, bot_response):
    cursor = db_connection.cursor()
    cursor.execute(
        "INSERT INTO conversation_logs (user_query, bot_response) VALUES (?, ?)",
        (user_query, bot_response)
    )
    db_connection.commit()

# Load the vector store from disk with OpenAI embeddings
embedding_model = OpenAIEmbeddings()
vector_store = Chroma(embedding_function=embedding_model, persist_directory=data_directory)

# Initialize the OpenAI LLM with the new class
openai_llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=1,
    max_tokens=1024
)

prompt_template = """
As a highly knowledgeable fashion assistant, your role is to accurately interpret fashion queries and 
provide responses using our specialized fashion database. Follow these directives to ensure optimal user interactions:
1. Precision in Answers: Respond solely with information directly relevant to the user's query from our fashion database. 
    Refrain from making assumptions or adding extraneous details.
2. Topic Relevance: Limit your expertise to specific fashion-related areas:
    - Fashion Trends
    - Personal Styling Advice
    - Seasonal Wardrobe Selections
    - Accessory Recommendations
3. Handling Off-topic Queries: For questions unrelated to fashion (e.g., general knowledge questions like "Why is the sky blue?"), 
    politely inform the user that the query is outside the chatbot’s scope and suggest redirecting to fashion-related inquiries.
4. Promoting Fashion Awareness: Craft responses that emphasize good fashion sense, aligning with the latest trends and 
    personalized style recommendations.
5. Contextual Accuracy: Ensure responses are directly related to the fashion query, utilizing only pertinent 
    information from our database.
6. Relevance Check: If a query does not align with our fashion database, guide the user to refine their 
    question or politely decline to provide an answer.
7. Avoiding Duplication: Ensure no response is repeated within the same interaction, maintaining uniqueness and 
    relevance to each user query.
8. Streamlined Communication: Eliminate any unnecessary comments or closing remarks from responses. Focus on
    delivering clear, concise, and direct answers.
9. Avoid Non-essential Sign-offs: Do not include any sign-offs like "Best regards" or "FashionBot" in responses.
10. One-time Use Phrases: Avoid using the same phrases multiple times within the same response. Each 
    sentence should be unique and contribute to the overall message without redundancy.

Fashion Query:
{context}

Question: {question}

Answer:
"""

custom_prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

rag_chain = RetrievalQA.from_chain_type(
    llm=openai_llm, 
    chain_type="stuff", 
    retriever=vector_store.as_retriever(top_k=3),
    chain_type_kwargs={"prompt": custom_prompt}
)

def get_response(question):
    # Ensure question is a string and not empty
    if not isinstance(question, str) or not question.strip():
        return "Please provide a valid question."
    
    try:
        result = rag_chain({"query": question})
        response_text = result["result"]
        answer_start = response_text.find("Answer:") + len("Answer:")
        answer = response_text[answer_start:].strip()
        # Log the conversation to the database
        log_conversation(question, answer)
        return answer
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Streamlit app
st.markdown(
    """
        <style>
            .appview-container .main .block-container {{
                padding-top: {padding_top}rem;
                padding-bottom: {padding_bottom}rem;
                }}
        </style>""".format(
        padding_top=1, padding_bottom=1
    ),
    unsafe_allow_html=True,
)

st.markdown("""<h3 style='text-align: left; color: white; padding-top: 35px; border-bottom: 3px solid red;'>
    Discover the AI Styling Recommendations 👗👠
</h3>""", unsafe_allow_html=True)

side_bar_message = """
Hi! 👋 I'm here to help you with your fashion choices. What would you like to know or explore?
\nHere are some areas you might be interested in:
1. **Fashion Trends** 👕👖
2. **Personal Styling Advice** 👢🧢
3. **Seasonal Wardrobe Selections** 🌞
4. **Accessory Recommendations** 💍

Feel free to ask me anything about fashion!
"""

with st.sidebar:
    st.title('🤖FashionBot: Your AI Style Companion')
    st.markdown(side_bar_message)

initial_message = """
    Hi there! I'm your FashionBot 🤖 
    Here are some questions you might ask me:\n
     🎀What are the top fashion trends this summer?\n
     🎀Can you suggest an outfit for a summer wedding?\n
     🎀What are some must-have accessories for winter season?\n
     🎀What type of shoes should I wear with a cocktail dress?\n
     🎀What's the best look for a professional photo shoot?
"""

if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": initial_message}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": initial_message}]

st.button('Clear Chat', on_click=clear_chat_history)

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Debugging: Check the received prompt
    print("Prompt received:", prompt)  # Debugging line

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Hold on, I'm fetching the latest fashion advice for you..."):
                response = get_response(prompt)
                placeholder = st.empty()
                full_response = response
                placeholder.markdown(full_response)
        message = {"role": "assistant", "content": full_response}
        st.session_state.messages.append(message)
