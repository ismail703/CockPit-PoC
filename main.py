import sqlite3
import operator
from typing import Annotated, TypedDict, Dict, Any, List, Literal
from pydantic import BaseModel, Field
import json
import os
from dotenv import load_dotenv

# LangChain / LangGraph Imports
from langgraph.graph import StateGraph, END, START  
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
import chromadb
from chromadb.utils import embedding_functions


# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================

DB_PATH = "cockpit.db"
CHROMA_PATH = "./chroma_db_store"
MODEL_NAME = "granite3.2:2b"
QWEN_MODEL = "qwen3:1.7b"

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Initialize LLM
# llm = ChatOpenAI(
#     model="meta-llama/llama-3.3-70b-instruct:free",
#     openai_api_key=api_key,
#     openai_api_base="https://openrouter.ai/api/v1",
#     default_headers={
#         "HTTP-Referer": "http://localhost:3000",
#         "X-Title": "llama",
#     }
# )

# qwen = ChatOpenAI(
#     model="qwen/qwen3-next-80b-a3b-instruct:free",
#     openai_api_key=api_key,
#     openai_api_base="https://openrouter.ai/api/v1",
#     default_headers={
#         "HTTP-Referer": "http://localhost:3000",
#         "X-Title": "qwen",
#     }
# )

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    groq_api_key=api_key,
    temperature=0, 
)

qwen = ChatGroq(
    model="qwen/qwen3-32b",
    groq_api_key=api_key,
    temperature=0, 
)

# llm = ChatOllama(model=MODEL_NAME, temperature=0)
# qwen = ChatOllama(model=QWEN_MODEL, temperature=0)

# Initialize ChromaDB Client
client = chromadb.PersistentClient(path=CHROMA_PATH)
ollama_ef = embedding_functions.OllamaEmbeddingFunction(
    url="http://localhost:11434/api/embeddings",
    model_name="qwen3-embedding:4b"
)

# Connect to your 4 existing collections
coll_schema = client.get_collection("telco_db_schema", embedding_function=ollama_ef)
coll_evidence = client.get_collection("telco_domain_evidence", embedding_function=ollama_ef)
coll_values = client.get_collection("telco_distinct_values", embedding_function=ollama_ef)
coll_examples = client.get_collection("sql_few_shot_examples", embedding_function=ollama_ef)

# ==========================================
# 2. DEFINE STATE & MODELS
# ==========================================

class VectorDBQueries(BaseModel):
    """Output model for generating targeted queries for each Vector DB"""
    schema_query: List[str] = Field(description="List of queries to find similar SQL patterns")
    knowledge_query: List[str] = Field(description="List of queries for domain rules")
    value_query: List[str] = Field(description="List of queries for specific data values")
    example_query: List[str] = Field(description="List of queries to find similar SQL patterns")

class SemanticCheckResult(BaseModel): 
    reasoning: str = Field(description="Explanation of why the SQL is correct or incorrect based on the user question.") 
    is_semantically_correct: bool = Field(description="True if the SQL perfectly matches the user intent. False if logic needs fixing.") 
    corrected_sql: str = Field(description="The fixed SQL query if incorrect. If correct, return the original SQL.") 


class AgentState(TypedDict):
    question: str                                    # User's question
    vect_queries: dict                               # Queries to fetch data from vector db each query with it's key
    db_results: Annotated[List[dict], operator.add]  # Results of all vector db
    sql_candidate: str                               # Generated SQL Query 
    is_sql_modified: bool = False                    # Flag to trigger the feedback loop
    query_result: str = ""                           # Stores the successful data
    retry_count: int = 0                             # Safety limit
    final_output: str                                # Response send to user                    # Error tracking

# ==========================================
# 3. DEFINE NODES
# ==========================================

def generate_vect_db_query(state: AgentState):
    """
    Node 1: Decompose the user's question into 4 specific search queries.
    """
    print(f"  [INFO] Generating Vector Queries for: '{state['question']}'   ")
    
    # Use the model to structure the output
    structured_llm = qwen.with_structured_output(VectorDBQueries)
    
    system_prompt = """
      You are a smart query decomposition assistant for a Telco Text-to-SQL system. Your task is to transform a user’s natural language question into three structured retrieval queries. Each query targets a different information source in a vector database to "ground" the intent before SQL generation. 
      You must always return these three fields:

        - schema_query 
        - knowledge_query
        - value_query
        - example_query

      1. Schema Query:
        Goal: Identify relevant tables and columns identifiers for Schema Alignment.
        Action: 
            Extract nouns that look like database objects (e.g., "active customer", "bill", "Recharge", "data"). 
            Identify entities that would logically represent a table or a specific column name in a telecom database.
            Extract words that can be used to describe the table

      2. Knowledge Query
        Goal: Retrieve business definitions, exact KPI names, and Evidence/Logic documentation.
        Action:
            Extract domain-specific terms, telecom KPIs, traffic types, or financial metrics.
            Focus on technical and business descriptors that clarify underlying logic (e.g., how "Churn" is calculated or what "National Mobile IAM" includes).

      3. Value Query
        Goal: Retrieve exact Data Vocabulary and string matches for categorical filters.
        Action:
            Extract proper nouns, capitalized words, offer names, plan names, product names, or alphanumeric identifiers.
            The goal is to prevent syntax errors by finding the exact DISTINCT value present in the database.

      4. Example Query
        Goal: Retrieve similar SQL templates for few-shot learning.
        Action: 
          Rewrite the user’s question so it can be used to fetch similar SQL queries by comparing it to existing questions.
          Do not write any SQL query or SQL Syntax
   
    FALLBACK RULE (MANDATORY):
    If a query type is not applicable, you must return the original user question for that field.
    Do not return empty strings.
    Do not return "None".
    Do not omit any field.

    EXAMPLES:

    USER QUESTION: Calculate the total count of recharges categorized as type '*3' performed during January 2024
    knowledge_query: ["type of recharge", "*3"]
    schema_query: ["recharge"]
    evidence_query: ["recharge types", "*3", "recharges", "recharge type *3 definition"] 
    example_query: ["the total count of recharges of type '*3' performed during January 2024"]

    USER QUESTION: What is the total number of active B2C customers on the iDar offer at the end of January 2026?
    schema_query: ["active B2C customers", "active customers", "customers"]
    knowledge_query: ["active B2C customers", iDar, B2C, "active customers at the end of January 2026", "the total number of active B2C customers on the iDar offer at the end of January 2026"]
    evidence_query: ["iDar", "B2C", "active customers"] 
    example_query: ["total number of active customers on the iDar offer", ]

    Always return all three fields to ensure 100% vocabulary alignment for the downstream LLM.
    """
    
    queries_object = structured_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=state['question'])
    ])
    
    print("schema: ", queries_object.schema_query)
    print("evidence: ", queries_object.knowledge_query)
    print("value: ", queries_object.value_query)
    print("example: ", queries_object.example_query)

    return {
        "vect_queries": {
            "schema": queries_object.schema_query,
            "evidence": queries_object.knowledge_query,
            "value": queries_object.value_query,
            "example": queries_object.example_query
        }
    }


# vect_queries = generate_vect_db_query({"question": "Provide the total count of active customers for each customer type"})
# vect_queries = generate_vect_db_query({"question": "I'd like to know the count of billing types for each customer type"})
# print(vect_queries)

def retrieve_schema(state: AgentState):
    """
    Node 2: Execute retrieval against the Vector Databases.
    Uses the list of strings generated by the Metadata Extractor.
    """
    queries = state['vect_queries']
    
    print(f"[INFO] Retrieving Schema & Metadata...")

    res_schema = []
    for q_text in queries['schema']:
        results = coll_schema.query(query_texts=[q_text], n_results=1)
        
        if results['documents'] and results['documents'][0]:
            doc_content = "\n".join(results['documents'][0])
            res_schema.append(doc_content)
    
    final_schema = "\n---\n".join(list(set(res_schema)))

    print("SCHEMA: \n", final_schema)
    
    return {"db_results": [final_schema]}

# db_schema = retrieve_schema(vect_queries)
# print(db_schema)

def retrieve_examples(state: AgentState):
    """
    Node: Execute retrieval for Few-Shot SQL examples.
    Iterates through example_query list to find relevant SQL patterns.
    """
    queries = state['vect_queries']
    print(f"  [INFO] Retrieving Examples...")

    search_terms = queries.get("example", state['question'])
    if isinstance(search_terms, list):
        search_term = search_terms[0]
    else:
        search_term = search_terms

    res_examples = coll_examples.query(query_texts=[search_term], n_results=2)
    
    examples_list = []
    if res_examples['metadatas'] and res_examples['metadatas'][0]:
        for meta, doc_text in zip(res_examples['metadatas'][0], res_examples['documents'][0]):
            sql_code = meta.get('query', 'No SQL found')
            examples_list.append(f"Question: {doc_text}\nSQL: {sql_code}")

    examples_txt = "\n---\n".join(examples_list)
    
    if not examples_txt:
        examples_txt = "No relevant SQL examples found."

    print(f"Examples retrieved ({len(examples_list)} snippets found)")

    return {"db_results": [examples_txt]}

# sql_example = retrieve_examples(vect_queries)
# print(sql_example)

def retrieve_evidence(state: AgentState):
    """
    Node 2: Execute retrieval for Business Logic and Evidence.
    Iterates through the list of knowledge queries.
    """
    queries = state['vect_queries']
    print(f"  [INFO] Retrieving Evidence...")
    
    search_terms = queries['evidence']
    
    unique_docs = set()
    
    for term in search_terms:
        res_evidence = coll_evidence.query(query_texts=[term], n_results=4)
        
        if res_evidence['documents'] and res_evidence['documents'][0]:
            for doc in res_evidence['documents'][0]:
                unique_docs.add(doc)

    evidence_txt = "\n\n".join(list(unique_docs))
    
    print(f"Evidence retrieved ({len(unique_docs)} unique snippets found)")

    return {"db_results": [evidence_txt]}

# evidence = retrieve_evidence(vect_queries)
# print(evidence)    


def retrieve_values(state: AgentState):
    """
    Node: Execute retrieval for specific Data Vocabulary / Categorical Values.
    Iterates through the value_query list to ground exact database spellings.
    """
    queries = state['vect_queries']
    print(f"  [INFO] Retrieving Values...")
    
    search_terms = queries["value"]
    
    unique_value_mappings = set()
    
    for term in search_terms:
        res_values = coll_values.query(query_texts=[term], n_results=6)

        if res_values.get('metadatas') and res_values['metadatas'][0]:
            for meta in res_values['metadatas'][0]:
                val = meta.get('value', 'Unknown')
                col = meta.get('column_name', 'Unknown')
                tbl = meta.get('table_name', 'Unknown')
                unique_value_mappings.add(f"Found Value: '{val}' in Table: {tbl}, Column: {col}")

    values_txt = "\n".join(list(unique_value_mappings))
    
    if not values_txt:
        values_txt = "No specific categorical value matches found."
    
    print("Values: ", values_txt)

    return {"db_results": [values_txt]}

# cell_values = retrieve_values(vect_queries)
# print(cell_values)


def generate_sql(state: AgentState):
    """
    Generate the SQL query using the retrieved context.
    """
    print("  [INFO] Generating SQL   ")
    
    full_context = "\n\n".join(state['db_results'])
    
    prompt = f"""You are an expert SQLite developer for a Telco company.
    
    RETRIEVED CONTEXT (Schema, Examples, Values, and Evidence):
    {full_context}

    USER QUESTION: "{state['question']}"
    
    INSTRUCTIONS:
    1. Write a valid SQLite query to answer the question.
    2. Use the provided context to identify correct tables, columns, and exact values.
    3. Return ONLY the SQL query. No markdown formatting, no explanations.
    4. Use valid filters and ensure that the values applied in the filters are correct by verifying them against the provided context.

    Find the right KPI to use to find result and use the right filters 
    """
    
    response = llm.invoke([prompt])
    cleaned_sql = response.content.replace("```sql", "").replace("```", "").strip()

    print("Generated SQL: ", cleaned_sql)
    
    return {"sql_candidate": cleaned_sql}

def syntax_checker(state: AgentState):
    """
    Node: Attempt to execute the SQL. If it fails, ask the LLM to fix it.
    """
    current_sql = state["sql_candidate"]
    retries = state.get("retry_count", 0)
    
    print(f"  [INFO] Checking Syntax (Attempt {retries + 1})")

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute(current_sql)
        result_data = cursor.fetchall()
        conn.close()

        print("   [Success] SQL executed successfully.")
        return {
            "query_result": str(result_data),
            "is_sql_modified": False,
            "retry_count": 0
        }

    except Exception as e:
        error_msg = str(e)
        print(f"   [Error] SQL failed: {error_msg}")
              
        if retries >= 3:
            print("   [Fail] Max retries reached.")
            return {
                "is_sql_modified": False,
                "query_result": f"Error: Failed after 3 attempts. Last error: {error_msg}",
                "retry_count": 0
            }
      
        full_context = "\n\n".join(state['db_results'])
        
        system_prompt = f"""You are a SQL Debugger. The user's SQLite query failed with an error. 
        Fix the query based ONLY on the error message provided. 
        Return ONLY the corrected SQL. No markdown formatting, no explanations.
        
        Use this CONTEXT (Schema, Values, and Evidence):
        {full_context}
        """
      
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Original Query: {current_sql}\nSQLite Error: {error_msg}")
        ]
      
        response = llm.invoke(messages)
        fixed_sql = response.content.replace("```sql", "").replace("```", "").strip()
     
        return {
            "sql_candidate": fixed_sql,
            "is_sql_modified": True,
            "retry_count": retries + 1
        }

def should_continue_syntax(state: AgentState) -> Literal["syntax_checker", "semantic_checker"]:
    """
    If the SQL was modified (meaning an error occurred and was fixed), 
    we loop back to check the NEW sql.
    """
    if state["is_sql_modified"]:
        return "syntax_checker"
    else:
        return "semantic_checker"        

def semantic_checker(state: AgentState):
    """
    Audit the SQL logic against the user's original intent.
    """
    print("  [INFO] Semantic Logic Review ")
    current_sql = state["sql_candidate"]
    original_question = state["question"]
    
    
    structured_llm = qwen.with_structured_output(SemanticCheckResult)
    
    full_context = "\n\n".join(state['db_results'])
  
    system_prompt = f"""
    You are a Senior SQL Analyst specializing in Telecom data auditing. 
    Your role is to perform a rigorous logical audit on generated SQL queries.

    CHECKLIST:

    1. Time Filtering: Ensure the date range (e.g., "last month") matches the user's intent exactly.
    2. Aggregation: Verify if the user asked for "average" vs "sum" vs "count".
    3. Joins & Segmentation: Ensure correct tables are joined for specific "Customer Types".

    4. KPI & Split: 
    - Verify the usage of the correct KPI name.
    - Distinguish between a 'Segment' (filtering a group) and a 'Split' (categorizing the output).
    5. Result Validation: ensure the result aligned with user question (should the query use the 'valeur_d1' column ?).
    6. Filters Validation: Ensure that the values applied in the filters (WHERE clause) are correct by verifying them against the provided context, and matching the appropriate data types

    CONSTRAINTS:
    - Do not invent column names or values. Use ONLY the provided Context (Schema, Evidence, and Values).
    - If the query is logically sound, return it as is.
    - If any logic is flawed, provide the corrected version in the 'corrected_sql' field.
    
    USE THE CONTEXT BELOW (Schema, Examples, Values, and Evidence):
    {full_context}
    """

    
    user_prompt = f"""
    User Question: "{original_question}"
    Candidate SQL: "{current_sql}"
    
    Evaluate if the SQL answers the question accurately.
    """
  
    result = structured_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])
  
    print(f"   [Reasoning]: {result.reasoning}")

    if result.is_semantically_correct:
        print("   [Success] Logic is sound.")
        return {
            "is_sql_modified": False 
        }
    else:       
        print("   [Warning] Logic error detected. Updating SQL.")
        print(f"  [INFO] Corrected SQL: {result.corrected_sql}")
        return {
            "sql_candidate": result.corrected_sql,
            "is_sql_modified": True,
            "retry_count": 0
        }

def check_semantic_modification(state: AgentState) -> Literal["syntax_checker", "format_result"]:
    if state.get("is_sql_modified"):
        print("   >> Looping back to Syntax Checker")
        return "syntax_checker"
    else:
        print("   >> Proceeding to Finish")
        return "format_result"

def format_result(state: AgentState):
    """
    Convert raw database results into a natural language response.
    """
    print("  [Node] Formatting Final Response")
    user_question = state["question"]
    raw_data = state["query_result"]
    
    if "Error:" in raw_data:
        return {"final_output": f"I'm sorry, I encountered an issue while processing your request: {raw_data}"}

    system_prompt = """You are an expert Telco Data Analyst. Your goal is to answer the user's question using the provided data.

    INPUT CONTEXT:
    - User Question: The original question asked.
    - Data Result: Raw data from the database (JSON, List, or Tuples).

    INSTRUCTIONS:
    1. **Synthesize**: Convert the raw data into a natural, complete sentence. Do not just dump the data.
    2. **Format**: 
    - For lists of 1-3 items, join them with commas.
    - For lists of 4+ items, use bullet points for readability.
    - Ensure numbers are formatted correctly (e.g., add '$' for revenue, 'GB' for data).
    3. **Handling Empty Data**: If the result is empty or "[]", reply: "I checked the records, but I couldn't find any information matching that request."
    4. **Tone**: Professional, concise, and helpful. 
    5. **Restriction**: Never mention "SQL", "tuples", "JSON", or "database schema". Speak the user's language.
    """
  
    user_message = f"""
    User Question: "{user_question}"
    Database Result: {raw_data}  
    
    Please provide a final answer to the user based on the data above.
    """

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message)
    ])
    
    return {"final_output": response.content}        

workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("generate_vect_db_query", generate_vect_db_query)
workflow.add_node("schema_db", retrieve_schema)
workflow.add_node("example_db", retrieve_examples)
workflow.add_node("evidence_db", retrieve_evidence)
workflow.add_node("cell_value_db", retrieve_values)
workflow.add_node("generate_query", generate_sql)
workflow.add_node("syntax_checker", syntax_checker)
workflow.add_node("semantic_checker", semantic_checker)
workflow.add_node("format_result", format_result)

# Define Edges

# Start -> Generate
workflow.add_edge(START, "generate_vect_db_query")

# Fan-out (One to Many)
workflow.add_edge("generate_vect_db_query", "schema_db")
workflow.add_edge("generate_vect_db_query", "example_db")
workflow.add_edge("generate_vect_db_query", "evidence_db")
workflow.add_edge("generate_vect_db_query", "cell_value_db")

# Fan-in (Many to One)
workflow.add_edge("schema_db", "generate_query")
workflow.add_edge("example_db", "generate_query")
workflow.add_edge("evidence_db", "generate_query")
workflow.add_edge("cell_value_db", "generate_query")
workflow.add_edge("generate_query", "syntax_checker")

workflow.add_conditional_edges(
    "syntax_checker",
    should_continue_syntax,
    {
        "syntax_checker": "syntax_checker",
        "semantic_checker": "semantic_checker"
    }
)

workflow.add_conditional_edges(
    "semantic_checker",
    check_semantic_modification,
    {
        "syntax_checker": "syntax_checker",
        "format_result": "format_result"
    }
)

workflow.add_edge("format_result", END)

graph = workflow.compile()

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "1"}}

    result = graph.invoke({"question": "What is the total number of new B2B customers at the year of 2025?"}, config)

    print("SQL: ", result["sql_candidate"])
    print(f"SQL Result {result['query_result']}")
    print(f"Formatted Result {result['query_result']}")
