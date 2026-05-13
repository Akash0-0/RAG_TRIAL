from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os 
import sys
import json
sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

print(f"API Key loaded: {os.getenv('OPENROUTER_API_KEY') is not None}\n")

presistent_directory = "db/chroma_db"
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  

llm = ChatOpenAI(
    model="openrouter/free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    timeout=30,
    max_retries=2
)

db = Chroma(
    persist_directory=presistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

# MAIN EXE
original_query = "How much money does Tesla make?"
print(f"Original Query: {original_query}\n")

## Step 1: Generate query variations via manual JSON parsing
prompt = f"""Generate 5 variations of the following query that maintain the same intent but use different wording.
Original query: "{original_query}"

Return ONLY a JSON array of 5 strings, no explanation, no markdown, no extra text.
Example format: ["query1", "query2", "query3", "query4", "query5"]"""

print("Generating query variations...")
raw_response = llm.invoke(prompt)
raw_text = raw_response.content.strip()

# Strip markdown code fences if model wraps in ```json ... ```
if "```" in raw_text:
    raw_text = raw_text.split("```")[1]
    if raw_text.startswith("json"):
        raw_text = raw_text[4:]
    raw_text = raw_text.strip()

Query_Variations = json.loads(raw_text)

print("Generated Query Variations:")
for i, variation in enumerate(Query_Variations, 1):
    print(f"{i}. {variation}")

print("\n---\n")

## Step 2: Retrieve documents for each query variation
retriever = db.as_retriever(search_kwargs={"k": 5})
all_retrieval_results = []

for i, query in enumerate(Query_Variations, 1):
    print(f"\n=== RESULTS FOR QUERY {i}: {query} ===")
    
    docs = retriever.invoke(query)
    all_retrieval_results.append(docs)
    
    print(f"Retrieved {len(docs)} documents:\n")
    for j, doc in enumerate(docs, 1):
        print(f"Document {j}:")
        print(f"{doc.page_content[:150]}...\n")
    
    print("-" * 50)

# Deduplicate
combined_docs = [doc for sublist in all_retrieval_results for doc in sublist]
seen = set()
unique_docs = []
for doc in combined_docs:
    if doc.page_content not in seen:
        seen.add(doc.page_content)
        unique_docs.append(doc)

print("\n" + "="*60)
print("Multi-Query Retrieval Complete!")
print(f"Total unique documents retrieved: {len(unique_docs)}")