#!/usr/bin/env python3
"""
Embed documents from local RAG folder using nomic-embed-text via Ollama
Saves embeddings to Chroma vectorstore
"""

import os
import glob
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
RAG_DIR    = "/home/soo/Work/mom-i/senior/rag"
CHROMA_DIR = "/home/soo/Work/mom-i/senior/chroma_db"
EMBED_MODEL = "nomic-embed-text"
OLLAMA_URL  = "http://localhost:11434"

CHUNK_SIZE    = 512
CHUNK_OVERLAP = 64

# ─────────────────────────────────────────
# STEP 1: LOAD DOCUMENTS
# ─────────────────────────────────────────
def load_documents():
    docs = []
    loaded_files = []

    # .txt files
    txt_files = glob.glob(os.path.join(RAG_DIR, "**/*.txt"), recursive=True)
    for f in txt_files:
        try:
            loader = TextLoader(f, encoding="utf-8")
            loaded = loader.load()
            docs.extend(loaded)
            loaded_files.append(f)
            print(f"  ✅ txt : {f}")
        except Exception as e:
            print(f"  ❌ txt failed: {f} → {e}")

    # .pdf files
    pdf_files = glob.glob(os.path.join(RAG_DIR, "**/*.pdf"), recursive=True)
    for f in pdf_files:
        try:
            loader = PyPDFLoader(f)
            loaded = loader.load()
            docs.extend(loaded)
            loaded_files.append(f)
            print(f"  ✅ pdf : {f}")
        except Exception as e:
            print(f"  ❌ pdf failed: {f} → {e}")

    # .json files
    json_files = glob.glob(os.path.join(RAG_DIR, "**/*.json"), recursive=True)
    for f in json_files:
        try:
            loader = TextLoader(f, encoding="utf-8")
            loaded = loader.load()
            docs.extend(loaded)
            loaded_files.append(f)
            print(f"  ✅ json: {f}")
        except Exception as e:
            print(f"  ❌ json failed: {f} → {e}")

    # .md files
    md_files = glob.glob(os.path.join(RAG_DIR, "**/*.md"), recursive=True)
    for f in md_files:
        try:
            loader = TextLoader(f, encoding="utf-8")
            loaded = loader.load()
            docs.extend(loaded)
            loaded_files.append(f)
            print(f"  ✅ md  : {f}")
        except Exception as e:
            print(f"  ❌ md failed: {f} → {e}")

    print(f"\n📄 Total loaded: {len(docs)} documents from {len(loaded_files)} files")
    return docs


# ─────────────────────────────────────────
# STEP 2: SPLIT INTO CHUNKS
# ─────────────────────────────────────────
def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(docs)
    print(f"✂️  Split into {len(chunks)} chunks "
          f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


# ─────────────────────────────────────────
# STEP 3: EMBED + SAVE TO CHROMA
# ─────────────────────────────────────────
def embed_and_save(chunks):
    print(f"\n🔢 Embedding with [{EMBED_MODEL}] via Ollama...")
    print(f"   This may take a while depending on document size...\n")

    embeddings = OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url=OLLAMA_URL
    )

    # Embed in batches to show progress
    BATCH_SIZE = 50
    total = len(chunks)

    if total <= BATCH_SIZE:
        # Small enough — embed all at once
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_DIR
        )
    else:
        # Large — embed in batches
        vectorstore = None
        for i in range(0, total, BATCH_SIZE):
            batch = chunks[i:i + BATCH_SIZE]
            pct = min(i + BATCH_SIZE, total)
            print(f"  ⏳ Embedding batch {pct}/{total}...")

            if vectorstore is None:
                vectorstore = Chroma.from_documents(
                    documents=batch,
                    embedding=embeddings,
                    persist_directory=CHROMA_DIR
                )
            else:
                vectorstore.add_documents(batch)

    vectorstore.persist()
    print(f"\n✅ Embeddings saved to: {CHROMA_DIR}")
    print(f"✅ Total vectors stored: {total}")
    return vectorstore


# ─────────────────────────────────────────
# STEP 4: VERIFY
# ─────────────────────────────────────────
def verify(vectorstore):
    print(f"\n🔍 Verifying vectorstore with test query...")
    test_query = "sleep quality"
    results = vectorstore.similarity_search(test_query, k=3)
    print(f"   Query: '{test_query}'")
    print(f"   Found {len(results)} results:")
    for i, doc in enumerate(results):
        preview = doc.page_content[:100].replace("\n", " ")
        print(f"   [{i+1}] {preview}...")
    print("\n✅ Vectorstore is working correctly!")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def main():
    print("=" * 55)
    print("  nomic-embed-text Document Embedder")
    print(f"  RAG DIR   : {RAG_DIR}")
    print(f"  CHROMA DIR: {CHROMA_DIR}")
    print(f"  MODEL     : {EMBED_MODEL}")
    print("=" * 55)

    # Check RAG directory exists
    if not os.path.exists(RAG_DIR):
        print(f"❌ RAG directory not found: {RAG_DIR}")
        return

    # Check if vectorstore already exists
    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        print(f"\n⚠️  Chroma DB already exists at: {CHROMA_DIR}")
        ans = input("   Rebuild from scratch? (y/n): ").strip().lower()
        if ans == "y":
            import shutil
            shutil.rmtree(CHROMA_DIR)
            print("   🗑️  Old vectorstore deleted")
        else:
            print("   ✅ Using existing vectorstore")
            embeddings = OllamaEmbeddings(
                model=EMBED_MODEL,
                base_url=OLLAMA_URL
            )
            vectorstore = Chroma(
                persist_directory=CHROMA_DIR,
                embedding_function=embeddings
            )
            verify(vectorstore)
            return

    # Load → Split → Embed → Save
    print(f"\n📂 Loading documents from: {RAG_DIR}\n")
    docs   = load_documents()

    if not docs:
        print("❌ No documents found! Check your RAG directory.")
        return

    chunks = split_documents(docs)
    vectorstore = embed_and_save(chunks)
    verify(vectorstore)

    print("\n🎉 Done! Ready to use in RAG pipeline.")


if __name__ == "__main__":
    main()
