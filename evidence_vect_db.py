import chromadb
import os
import json
from chromadb.utils import embedding_functions

# ==========================================
# 1. READ DOMAIN KNOWLEDGE FROM JSON
# ==========================================

JSON_FILE_PATH = "evidence.json"

if not os.path.exists(JSON_FILE_PATH):
    print(f"❌ Error: {JSON_FILE_PATH} not found.")
    exit()

with open(JSON_FILE_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# Ensure it's a list of strings
if not isinstance(data, list):
    print("❌ JSON file must contain a list of strings.")
    exit()

# Clean & filter empty strings
domain_knowledge = [str(item).strip() for item in data if str(item).strip()]

print(f"📄 Loaded {len(domain_knowledge)} evidence entries.")

# ==========================================
# 2. SETUP CHROMADB & OLLAMA
# ==========================================

DB_PATH = "./chroma_db_store"
COLLECTION_NAME = "telco_domain_evidence"

client = chromadb.PersistentClient(path=DB_PATH)

ollama_ef = embedding_functions.OllamaEmbeddingFunction(
    url="http://localhost:11434/api/embeddings",
    model_name="qwen3-embedding:4b"
)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=ollama_ef
)

# ==========================================
# 3. POPULATE DATABASE
# ==========================================

ids = []
documents = []

for idx, fact in enumerate(domain_knowledge):
    ids.append(f"ev_{idx}")
    documents.append(fact)

if ids:
    BATCH_SIZE = 5

    for i in range(0, len(ids), BATCH_SIZE):
        batch_ids = ids[i:i+BATCH_SIZE]
        batch_docs = documents[i:i+BATCH_SIZE]

        print(f"⏳ Inserting batch {i//BATCH_SIZE + 1}")

        collection.add(
            ids=batch_ids,
            documents=batch_docs
        )
    print("✅ All batches inserted successfully.")
else:
    print("⚠️ No valid data found in JSON file.")

# ==========================================
# 4. TEST RETRIEVAL
# ==========================================

test_question = "Churn"
print(f"\n🔍 Testing Retrieval for: '{test_question}'")

results = collection.query(
    query_texts=[test_question],
    n_results=2
)

if results['documents'] and results['documents'][0]:
    for i, matched_evidence in enumerate(results['documents'][0]):
        print(f"RETRIEVED FACT {i+1}: {matched_evidence}")
else:
    print("No match found.")