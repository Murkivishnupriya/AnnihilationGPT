# ---------------------------------------------------
# AmbedkarGPT – Robust RAG with ChromaDB + Llama 3
# ---------------------------------------------------
import os
import sys
from sentence_transformers import SentenceTransformer
import chromadb
import ollama

# --------------- CONFIGURATION ---------------------
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_DB_PATH = "database"
COLLECTION_NAME = "speech_collection"

# cached embedding model
_embedding_model = None


# --------------- EMBEDDING MODEL ---------------------
def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        print("Loading embedding model (this may take a moment)...")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


# --------------- STEP 1: LOAD TEXT -------------------
def load_speech():
    if not os.path.exists("speech.txt"):
        print("ERROR: speech.txt not found in the folder.")
        sys.exit(1)

    with open("speech.txt", "r", encoding="utf-8") as f:
        return f.read().strip()


# --------------- STEP 2: CHUNK TEXT -------------------
def chunk_text(text, chunk_size=100):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size]).strip()
        if chunk:
            chunks.append(chunk)

    return chunks


# --------------- STEP 3: EMBEDDINGS -------------------
def create_embeddings(chunks):
    model = get_embedding_model()
    embeddings = model.encode(chunks)
    # ensure pure python lists for Chroma
    try:
        embeddings = [e.tolist() if hasattr(e, "tolist") else list(e) for e in embeddings]
    except Exception:
        # fallback: leave as returned (Chroma often accepts numpy arrays too)
        pass
    return embeddings


# --------------- STEP 4: SAVE TO CHROMADB -------------
def save_to_chroma(chunks, embeddings):
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    ids = [f"chunk_{i}" for i in range(len(chunks))]

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids
    )

    print(f"\nSaved {len(chunks)} chunks into ChromaDB!")


# --------------- RAG: RETRIEVE TOP K CHUNKS ----------
def retrieve_context(query, top_k=3):
    print("\n=== Retrieving context from DB ===")

    model = get_embedding_model()
    q_emb = model.encode([query])[0]
    # convert to pure list if numpy
    try:
        q_emb = q_emb.tolist()
    except Exception:
        try:
            q_emb = list(q_emb)
        except Exception:
            pass

    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_collection(COLLECTION_NAME)

    # chroma expects a list-of-query-embeddings
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k
    )

    docs_list = results.get("documents", [[]])

    if not docs_list or not docs_list[0]:
        print("No documents found for the query.")
        return ""

    chunks = docs_list[0]
    combined = "\n\n---\n\n".join(chunks)

    print("\n=== Retrieved Chunks Preview ===")
    for i, c in enumerate(chunks, 1):
        preview = c.replace("\n", " ")[:300]
        print(f"\n--- Chunk {i} ---\n{preview}...")

    return combined


# ----------- LLM ANSWERING VIA OLLAMA ----------------
def generate_llm_answer(question, context):
    if not context.strip():
        return "I could not find relevant information in the provided context."

    prompt = f"""
You are AmbedkarGPT — answer ONLY using the context below.

Context:
\"\"\" 
{context}
\"\"\" 

Question: {question}

Instructions:
- Answer in 2–5 sentences.
- Use ONLY the given context.
- If the information is not in the context, say: "Not found in the provided context."
"""

    try:
        response = ollama.generate(
            model="llama3",
            prompt=prompt,
            stream=False
        )
    except Exception as e:
        return f"[LLM error] Could not generate answer: {e}"

    # ----------------------------
    # Robust extraction of text
    # ----------------------------
    text = ""

    # if dict-like
    if isinstance(response, dict):
        for k in ("response", "output", "message", "text", "data"):
            if k in response and response[k]:
                text = response[k]
                break

    # if nothing found, stringify
    if not text:
        text = str(response)

    # If the returned string contains a response=... style debug blob,
    # extract the piece after response= and before next metadata token.
    if "response=" in text:
        # take after first occurrence
        text = text.split("response=", 1)[1]

    # cut at common metadata markers (defensive)
    for marker in [" thinking=", " context=[", " logprobs=", " model=", " created_at=", " done_reason=", " total_duration=", " eval_count="]:
        if marker in text:
            text = text.split(marker, 1)[0]

    # final cleanup
    if isinstance(text, (list, dict)):
        text = str(text)

    text = text.strip()
    # remove surrounding quotes if present
    if (text.startswith("'") and text.endswith("'")) or (text.startswith('"') and text.endswith('"')):
        text = text[1:-1].strip()

    # last safe replace
    text = text.replace("'", "").replace('"', "").strip()

    # if still empty, fall back to a message
    if not text:
        return "Unable to parse LLM response (empty)."

    return text


# ------------------------ MAIN LOOP ---------------------
def main():
    print("\nDo you want to build the database again (y/n)?")
    choice = input("> ").strip().lower()

    if choice == "y":
        print("\n=== Loading Speech ===")
        text = load_speech()

        print("\n=== Splitting into Chunks ===")
        chunks = chunk_text(text, chunk_size=50)
        print("Total chunks:", len(chunks))

        print("\n=== Creating Embeddings ===")
        embeddings = create_embeddings(chunks)
        # attempt to show shape-like info
        try:
            print("Embedding count:", len(embeddings))
        except Exception:
            pass

        print("\n=== Saving to ChromaDB ===")
        save_to_chroma(chunks, embeddings)

    print("\nDatabase is ready. Ask anything about the speech!")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            question = input("Your question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not question:
            continue

        if question.lower() == "exit":
            print("Goodbye!")
            break

        context = retrieve_context(question, top_k=3)
        answer = generate_llm_answer(question, context)

        print("\nAmbedkarGPT:")
        print(answer)
        print("\n------------------------------------\n")


if __name__ == "__main__":
    main()
