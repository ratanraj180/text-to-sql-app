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

# LLM instances will be created on-demand inside the get_llm() function

# ------------------ UI ------------------
def get_llminfo():
    st.sidebar.header("Options", divider="rainbow")
    model = st.sidebar.radio(
        "Choose LLM:",
        ("gemini-2.5-flash", "gemini-2.5-pro", "llama3", "groq")
    )
    return model

# ------------------ DB ------------------
def connectDatabase(username, port, host, password, database):
    uri = f"mysql+mysqlconnector://{username}:{password}@{host}:{port}/{database}"
    st.session_state.db = SQLDatabase.from_uri(uri)

def getDatabaseSchema():
    return st.session_state.db.get_table_info()

def get_llm(model_name):
    if model_name == "gemini-2.5-flash":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment")
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, google_api_key=api_key)
    
    elif model_name == "gemini-2.5-pro":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment")
        return ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0, google_api_key=api_key)
    
    elif model_name == "llama3":
        return ChatOllama(model="llama3", temperature=0)
    
    elif model_name == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment")
        return ChatGroq(model="llama-3.1-8b-instant", temperature=0, api_key=api_key)

# ------------------ SQL GENERATION ------------------
def getQueryFromLLM(question, model):

    prompt = ChatPromptTemplate.from_template("""
You are an expert SQL assistant.

Schema:
{schema}

Convert the question into ONLY SQL query.

Question: {question}
SQL:
""")

    llm = get_llm(model)

    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({
        "question": question,
        "schema": getDatabaseSchema()
    })

    # clean SQL output
    return response.replace("```sql", "").replace("```", "").strip()

# ------------------ RUN SQL ------------------
def runQuery(query):
    try:
        return st.session_state.db.run(query)
    except Exception as e:
        return f"SQL Error: {e}"

# ------------------ NATURAL RESPONSE ------------------
def getResponseForQueryResult(question, query, model, result):

    prompt2 = ChatPromptTemplate.from_template("""
Schema:
{schema}

Question: {question}
SQL: {query}
Result: {result}

Explain the result in simple English.
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
st.set_page_config(page_title="Text to SQL", page_icon="🧠", layout="centered")
st.title("🧠 Ask Questions To Your MySQL Database")

model = get_llminfo()

with st.sidebar:
    st.header("Connect Database")
    host = st.text_input("Host", "localhost")
    port = st.text_input("Port", "3306")
    user = st.text_input("Username", "root")
    password = st.text_input("Password", type="password")
    database = st.text_input("Database", "rag_test")

    if st.button("Connect"):
        connectDatabase(user, port, host, password, database)
        st.success("Database Connected")

# ------------------ CHAT ------------------
question = st.chat_input("Ask your database...")

if "chat" not in st.session_state:
    st.session_state.chat = []

if question:
    if "db" not in st.session_state:
        st.error("Please connect database first")
    else:
        sql = getQueryFromLLM(question, model)
        result = runQuery(sql)
        answer = getResponseForQueryResult(question, sql, model, result)

        st.session_state.chat.append(("user", question))
        st.session_state.chat.append(("assistant", answer))

# display chat
for role, msg in st.session_state.chat:
    st.chat_message(role).markdown(msg)