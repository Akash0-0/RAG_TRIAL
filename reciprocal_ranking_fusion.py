from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os 
import sys
import json
sys.stdout.reconfigure(encoding='utf-8')
from pydantic import BaseModel
from typing import List
from collections import defaultdict

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


# ──────────────────────────────────────────────────────────────────
# Step 3: Apply Reciprocal Rank Fusion (RRF)
# ──────────────────────────────────────────────────────────────────

def reciprocal_rank_fusion(chunk_lists, k=60, verbose=True):

    if verbose:
        print("\n" + "="*60)
        print("APPLYING RECIPROCAL RANK FUSION")
        print("="*60)
        print(f"\nUsing k={k}")
        print("Calculating RRF scores...\n")
    
    # Data structures for RRF calculation
    rrf_scores = defaultdict(float)  # Will store: {chunk_content: rrf_score}
    all_unique_chunks = {}  # Will store: {chunk_content: actual_chunk_object}
    
    # For verbose output - track chunk IDs
    chunk_id_map = {}
    chunk_counter = 1
    
    # Go through each retrieval result
    for query_idx, chunks in enumerate(chunk_lists, 1):
        if verbose:
            print(f"Processing Query {query_idx} results:")
        
        # Go through each chunk in this query's results
        for position, chunk in enumerate(chunks, 1):  # position is 1-indexed
            # Use chunk content as unique identifier
            chunk_content = chunk.page_content
            
            # Assign a simple ID if we haven't seen this chunk before
            if chunk_content not in chunk_id_map:
                chunk_id_map[chunk_content] = f"Chunk_{chunk_counter}"
                chunk_counter += 1
            
            chunk_id = chunk_id_map[chunk_content]
            
            # Store the chunk object (in case we haven't seen it before)
            all_unique_chunks[chunk_content] = chunk
            
            # Calculate position score: 1/(k + position)
            position_score = 1 / (k + position)
            
            # Add to RRF score
            rrf_scores[chunk_content] += position_score
            
            if verbose:
                print(f"  Position {position}: {chunk_id} +{position_score:.4f} (running total: {rrf_scores[chunk_content]:.4f})")
                print(f"    Preview: {chunk_content[:80]}...")
        
        if verbose:
            print()
    
    # Sort chunks by RRF score (highest first)
    sorted_chunks = sorted(
        [(all_unique_chunks[chunk_content], score) for chunk_content, score in rrf_scores.items()],
        key=lambda x: x[1],  # Sort by RRF score
        reverse=True  # Highest scores first
    )
    
    if verbose:
        print(f" RRF Complete! Processed {len(sorted_chunks)} unique chunks from {len(chunk_lists)} queries.")
    
    return sorted_chunks

# Apply RRF to our retrieval results
fused_results = reciprocal_rank_fusion(all_retrieval_results, k=60, verbose=True)

# ──────────────────────────────────────────────────────────────────
# Step 4: Display Final Fused Results
# ──────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("FINAL RRF RANKING")
print("="*60)

print(f"\nTop {min(10, len(fused_results))} documents after RRF fusion:\n")

for rank, (doc, rrf_score) in enumerate(fused_results[:10], 1):
    print(f" RANK {rank} (RRF Score: {rrf_score:.4f})")
    print(f"{doc.page_content[:200]}...")
    print("-" * 50)

print(f"\n RRF Complete! Fused {len(fused_results)} unique documents from {len(Query_Variations)} query variations.")
print("\n Key benefits:")
print("   • Documents appearing in multiple queries get boosted scores")
print("   • Higher positions contribute more to the final score") 
print("   • Balanced fusion using k=60 for gentle position penalties")

# ──────────────────────────────────────────────────────────────────
# Optional: Quick Usage Examples
# ──────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("USAGE EXAMPLES")
print("="*60)