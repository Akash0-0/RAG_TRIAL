from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter

tesla_text = """Tesla's Q3 Results

Tesla reported record revenue of $25.2B in Q3 2024.

Model Y Performance

The Model Y became the best-selling vehicle globally, with 350,000 units sold.

Production Challenges

Supply chain issues caused a 12% increase in production costs.

This is one very long paragraph that definitely exceeds our 100 character limit and has no double newlines inside it whatsoever making it impossible to split properly."""

##splitter1 = CharacterTextSplitter(
#    chunk_size=100,
#    chunk_overlap=0,
 #   separator=" " # ( "\n" or "," or "." or " ")
#)
#
#chunks1 = splitter1.split_text(tesla_text)
#for i, chunk in enumerate(chunks1, 1):
#    print(f"--- Chunk {i} ---")
#    print(chunk,len(chunk))


splitter2 = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separators=["\n\n", "\n", ". ", " ", ""] #multi sepratror sorted by priority
)

chunks2 = splitter2.split_text(tesla_text)
for i ,chunk in enumerate(chunks2, 1):
    print(f"--- Chunk {i} ---")
    print(chunk,"\nlength:", len(chunk))