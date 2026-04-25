import streamlit as st
import os
from dotenv import load_dotenv

from langchain_community.chat_models import ChatOllama
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

load_dotenv()

# ------------------ DB INITIALIZATION ------------------
# We use SQLite for deployment compatibility
DB_PATH = "rag_test.db"
if not os.path.exists(DB_PATH):
    st.error(f"Error: {DB_PATH} not found. Please ensure the database file is in the repository.")

def get_db():
    if "db" not in st.session_state:
        st.session_state.db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")
    return st.session_state.db

def getDatabaseSchema():
    return get_db().get_table_info()

# ------------------ LLM FACTORY ------------------
def get_llm(model_name):
    if model_name == "gemini-2.5-flash":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            st.error("GOOGLE_API_KEY not found. Please set it in Streamlit Secrets.")
            st.stop()
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key=api_key)
    
    elif model_name == "gemini-2.5-pro":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            st.error("GOOGLE_API_KEY not found. Please set it in Streamlit Secrets.")
            st.stop()
        return ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0, google_api_key=api_key)
    
    elif model_name == "llama3":
        return ChatOllama(model="llama3", temperature=0)
    
    elif model_name == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            st.error("GROQ_API_KEY not found. Please set it in Streamlit Secrets.")
            st.stop()
        return ChatGroq(model="llama-3.1-8b-instant", temperature=0, api_key=api_key)

# ------------------ CORE LOGIC ------------------
def getQueryFromLLM(question, model):
    prompt = ChatPromptTemplate.from_template("""
You are an expert SQL assistant. System: SQLite.

Schema:
{schema}

Convert the user question into a valid SQLite query. Return ONLY the SQL code, no markdown blocks, no explanations.

Question: {question}
SQL:
""")

    llm = get_llm(model)
    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({
        "question": question,
        "schema": getDatabaseSchema()
    })

    return response.replace("```sql", "").replace("```", "").strip()

def runQuery(query):
    try:
        return get_db().run(query)
    except Exception as e:
        return f"SQL Error: {e}"

def getResponseForQueryResult(question, query, model, result):
    prompt2 = ChatPromptTemplate.from_template("""
You are a helpful data analyst.

Schema:
{schema}

User Question: {question}
Generated SQL: {query}
Database Result: {result}

Provide a concise, friendly explanation of the result in plain English.
""")

    llm = get_llm(model)
    chain2 = prompt2 | llm

    response = chain2.invoke({
        "schema": getDatabaseSchema(),
        "question": question,
        "query": query,
        "result": result
    })

    return response.content

# ------------------ APP UI ------------------
st.set_page_config(page_title="DataInsight SQL", page_icon="📊", layout="wide")

# Custom Styling
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stChatMessage {
        border-radius: 15px;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 DataInsight: Natural Language to SQL")
st.markdown("Ask questions about your music database (Tracks, Albums, Artists, etc.) in plain English.")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    model = st.radio(
        "Select AI Model:",
        ("gemini-2.5-flash", "gemini-2.5-pro", "groq"),
        index=0
    )
    
    st.divider()
    st.info("Database: `Chinook (SQLite)`")
    if st.button("View Schema Info"):
        st.code(getDatabaseSchema())

# ------------------ CHAT INTERFACE ------------------
if "chat" not in st.session_state:
    st.session_state.chat = []

for role, msg in st.session_state.chat:
    st.chat_message(role).markdown(msg)

if question := st.chat_input("Ex: 'Who are the top 5 artists by number of tracks?'"):
    # Display user message
    st.chat_message("user").markdown(question)
    st.session_state.chat.append(("user", question))

    with st.spinner("Analyzing data..."):
        try:
            # 1. Generate SQL
            sql_query = getQueryFromLLM(question, model)
            
            # 2. Run Query
            query_result = runQuery(sql_query)
            
            # 3. Generate Natural Language Response
            final_answer = getResponseForQueryResult(question, sql_query, model, query_result)
            
            # Display Assistant response
            with st.chat_message("assistant"):
                st.markdown(final_answer)
                with st.expander("View Generated SQL"):
                    st.code(sql_query, language="sql")
                if "SQL Error" not in str(query_result):
                    with st.expander("View Raw Data"):
                        st.write(query_result)
            
            st.session_state.chat.append(("assistant", final_answer))
            
        except Exception as e:
            st.error(f"Something went wrong: {e}")