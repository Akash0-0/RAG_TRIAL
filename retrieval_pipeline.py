from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import os

load_dotenv()

persistent_directory = "db/chroma_db"

# Load the same local embedding model used during ingestion
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

# Search for relevant documents
query = "How much did Microsoft pay to acquire GitHub?"

retriever = db.as_retriever(search_kwargs={"k": 5})

# retriever = db.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={
#         "k": 5,
#         "score_threshold": 0.3  # Only return chunks with cosine similarity >= 0.3
#     }
# )

relevant_docs = retriever.invoke(query)

print(f"User Query: {query}")
# Display results
print("--- Context ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")


import os
print("KEY:", os.getenv("OPENROUTER_API_KEY"))
# Query for LLM
combined_input = f"""Based on the following documents, please answer this question: {query}

Documents:
{chr(10).join([doc.page_content for doc in relevant_docs])}

Please provide a concise answer based on the information from the documents. If the answer is not explicitly stated, please infer it based on the context provided.
"""

# Create an LLM API inference via OpenRouter
# OpenRouter is fully compatible with the OpenAI SDK format —
# just point base_url at OpenRouter and pass your OPENROUTER_API_KEY
model = ChatOpenAI(
    model="openrouter/free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

# Define the messages for LLM
messages = [
    SystemMessage(content="You are a helpful assistant that answers questions based on provided documents."),
    HumanMessage(content=combined_input)
]

# Invoking the LLM API
result = model.invoke(messages)

# Display the answer
print("\n--- LLM Answer ---")

print("FULL RESULT:")
print(result)

print("\nAnswer:")
print(result.content)


# Synthetic Questions:

# 1. "What was NVIDIA's first graphics accelerator called?"
# 2. "Which company did NVIDIA acquire to enter the mobile processor market?"
# 3. "What was Microsoft's first hardware product release?"
# 4. "How much did Microsoft pay to acquire GitHub?"
# 5. "In what year did Tesla begin production of the Roadster?"
# 6. "Who succeeded Ze'ev Drori as CEO in October 2008?"
# 7. "What was the name of the autonomous spaceport drone ship that achieved the first successful sea landing?"
# 8. "What was the original name of Microsoft before it became Microsoft?"
