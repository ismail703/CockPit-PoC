import json
import os
import chromadb
from chromadb.utils import embedding_functions

# ==========================================
# 1. CONFIGURATION
# ==========================================

JSON_DATA_PATH = "db_values.json" 
CHROMA_DB_PATH = "./chroma_db_store"
COLLECTION_NAME = "telco_distinct_values"

# ==========================================
# 2. SETUP CHROMADB & OLLAMA
# ==========================================

client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

ollama_ef = embedding_functions.OllamaEmbeddingFunction(
    url="http://localhost:11434/api/embeddings",
    model_name="qwen3-embedding:4b"
)

# Optional: Clear old collection
try:
    client.delete_collection(name=COLLECTION_NAME)
    print(f"🗑️ Deleted old collection '{COLLECTION_NAME}'")
except:
    pass

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=ollama_ef
)

# ==========================================
# 3. EXTRACTION FROM JSON & INDEXING
# ==========================================

if not os.path.exists(JSON_DATA_PATH):
    print(f"❌ Error: {JSON_DATA_PATH} not found.")
    exit()

with open(JSON_DATA_PATH, "r", encoding="utf-8") as f:
    tables_data = json.load(f)

ids = []
documents = []
metadatas = []
global_counter = 0

for table in tables_data:
    table_name = table.get("table_name", "").strip()
    columns = table.get("columns", [])

    print(f"\n📊 Processing table: {table_name}")

    for col in columns:
        col_name = col.get("column_name", "").strip()
        values = col.get("values", [])

        if not col_name:
            continue

        print(f"   ➤ Column '{col_name}' → {len(values)} values")

        for val in values:
            if val is None or not str(val).strip():
                continue

            val_str = str(val).strip()

            ids.append(f"val_{global_counter}")

            # Better document representation (important for embeddings quality 🔥)
            documents.append(
                f"{val_str}"
            )

            metadatas.append({
                "table_name": table_name,
                "column_name": col_name,
                "value": val_str
            })

            global_counter += 1

# ==========================================
# 4. STORE IN VECTOR DB
# ==========================================

if ids:
    print(f"\n⏳ Embedding and storing {len(ids)} values...")
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )
    print("✅ Value Indexing Complete.")
else:
    print("⚠️ No values found in JSON to index.")

# ==========================================
# 5. TEST RETRIEVAL (Sanity Check)
# ==========================================

test_term = "GAS"
print(f"\n🔍 Testing Retrieval for user term: '{test_term}'")

results = collection.query(
    query_texts=[test_term],
    n_results=5
)

if results['ids'] and len(results['ids'][0]) > 0:
    for i in range(len(results['ids'][0])):
        match_doc = results['documents'][0][i]
        match_meta = results['metadatas'][0][i]

        print(f"\n--- Match {i+1} ---")
        print(f"Document       : {match_doc}")
        print(f"Table          : {match_meta['table_name']}")
        print(f"Column         : {match_meta['column_name']}")
        print(f"Value          : {match_meta['value']}")
else:
    print("No match found.")