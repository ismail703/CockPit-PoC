import chromadb
import json
import os
from chromadb.utils import embedding_functions

# ==========================================
# 1. LOAD DATA FROM JSON FILE
# ==========================================
JSON_FILE_PATH = "question-example.json" 

if not os.path.exists(JSON_FILE_PATH):
    print(f"❌ Error: {JSON_FILE_PATH} not found.")
    exit()

with open(JSON_FILE_PATH, "r") as f:
    examples_data = json.load(f)

# ==========================================
# 2. SETUP CHROMADB & OLLAMA
# ==========================================
DB_PATH = "./chroma_db_store"
client = chromadb.PersistentClient(path=DB_PATH)

ollama_ef = embedding_functions.OllamaEmbeddingFunction(
    url="http://localhost:11434/api/embeddings",
    model_name="qwen3-embedding:4b"
)

collection = client.get_or_create_collection(
    name="sql_few_shot_examples",
    embedding_function=ollama_ef
)

# ==========================================
# 3. POPULATE DATABASE (Description Removed)
# ==========================================
ids = []
documents = []
metadatas = []

for idx, ex in enumerate(examples_data):
    ids.append(f"ex_{idx}")
    documents.append(ex["question"])
    
    # Metadata now only contains the SQL query
    metadatas.append({
        "query": ex["sql"]
    })

collection.add(
    ids=ids,
    documents=documents,
    metadatas=metadatas
)

print(f"✅ Successfully stored {len(examples_data)} examples.")

# ==========================================
# 4. TEST RETRIEVAL
# ==========================================
new_user_question = "What is the total number of Revenue for each customer segment?"

results = collection.query(
    query_texts=[new_user_question],
    n_results=1
)

if results['metadatas']:
    print(f"\n🔍 Matched: {results['documents'][0][0]}")
    print(f"💻 SQL: {results['metadatas'][0][0]['query']}")