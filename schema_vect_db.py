import chromadb
from chromadb.utils import embedding_functions
import json
import os

# ==========================================
# 1️⃣ LOAD SCHEMA FROM JSON FILE
# ==========================================

JSON_SCHEMA_PATH = "db_schema.json"

if not os.path.exists(JSON_SCHEMA_PATH):
    print(f"❌ File {JSON_SCHEMA_PATH} not found.")
    exit()

with open(JSON_SCHEMA_PATH, "r", encoding="utf-8") as f:
    tables_json = json.load(f)

if not isinstance(tables_json, list):
    print("❌ JSON must contain a list of tables.")
    exit()

print(f"📄 Loaded {len(tables_json)} tables from JSON.")

# ==========================================
# 2️⃣ SETUP CHROMADB & OLLAMA
# ==========================================

client = chromadb.PersistentClient(path="./chroma_db_store")

ollama_ef = embedding_functions.OllamaEmbeddingFunction(
    url="http://localhost:11434/api/embeddings",
    model_name="qwen3-embedding:4b"
)

collection = client.get_or_create_collection(
    name="telco_db_schema",
    embedding_function=ollama_ef
)

# ==========================================
# 3️⃣ POPULATE VECTOR DB
# ==========================================

print("⏳ Generating embeddings and populating Vector DB...")

ids = []
documents = []
metadatas = []

for table in tables_json:

    table_name = table.get("table_name", "").strip()
    description = table.get("description", "").strip()
    columns = table.get("columns", [])
    foreign_keys = table.get("foreign_keys", [])

    if not table_name:
        continue

    # Unique ID per table
    ids.append(table_name)

    # Build column details
    col_details_list = []
    for col in columns:
        col_name = col.get("column_name", "").strip()
        col_desc = col.get("description", "").strip()
        col_type = col.get("datatype", "").strip()

        if col_name:
            col_details_list.append(
                f"{col_name} ({col_type}): {col_desc}"
            )

    col_details = ", ".join(col_details_list)

    # Text used for embedding (VERY IMPORTANT FOR NL2SQL QUALITY 🔥)
    text_content = (
        f"Table: {table_name}. "
        f"Description: {description}. "
        f"Columns: {col_details}. "
    )

    documents.append(text_content)

    # Metadata
    meta = {
        "table_name": table_name,
        "description": description,
        "columns": json.dumps(columns),
    }

    metadatas.append(meta)

# Use batching to avoid timeout
BATCH_SIZE = 1

for i in range(0, len(ids), BATCH_SIZE):
    collection.upsert(
        ids=ids[i:i+BATCH_SIZE],
        documents=documents[i:i+BATCH_SIZE],
        metadatas=metadatas[i:i+BATCH_SIZE]
    )

print(f"✅ Successfully stored {len(ids)} tables in Vector DB.")

# ==========================================
# 4️⃣ TEST QUERY
# ==========================================

query_text = "Customers activation"
print(f"\n🔍 Testing Query: '{query_text}'")

results = collection.query(
    query_texts=[query_text],
    n_results=3
)

if results["ids"] and results["ids"][0]:
    top_match_id = results["ids"][0][0]
    top_match_meta = results["metadatas"][0][0]

    print(f"🏆 Top Match Table: {top_match_id}")
    print(f"📄 Description: {top_match_meta['description']}")
else:
    print("No match found.")