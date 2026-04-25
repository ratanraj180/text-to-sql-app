import os
from dotenv import load_dotenv

from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

db = SQLDatabase.from_uri("mysql+mysqlconnector://root@localhost:3306/rag_test")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

question = "Give top 3 students sorted by marks in descending order"

prompt = ChatPromptTemplate.from_template("""
You are an expert SQL assistant.

Schema:
{schema}

Convert the question into ONLY SQL query.

Question: {question}
SQL:
""")

chain = prompt | llm | StrOutputParser()

sql = chain.invoke({
    "question": question,
    "schema": db.get_table_info()
})

sql = sql.replace("```sql", "").replace("```", "").strip()
print("GENERATED SQL:", sql)

try:
    result = db.run(sql)
except Exception as e:
    result = f"SQL Error: {e}"

print("RESULT:", result)

prompt2 = ChatPromptTemplate.from_template("""
Schema:
{schema}

Question: {question}
SQL: {query}
Result: {result}

Explain the result in simple English.
""")

chain2 = prompt2 | llm
response = chain2.invoke({
    "schema": db.get_table_info(),
    "question": question,
    "query": sql,
    "result": result
})

print("FINAL ANSWER:", response.content)
