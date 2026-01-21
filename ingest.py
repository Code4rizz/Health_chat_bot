import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

DATA_DIR = "data"
INDEX_DIR = "health_index"

docs = []

for file in os.listdir(DATA_DIR):
    if file.endswith(".txt"):
        loader = TextLoader(os.path.join(DATA_DIR, file), encoding="utf-8")
        loaded = loader.load()
        for d in loaded:
            d.metadata["source"] = file.replace(".txt", "")
        docs.extend(loaded)

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.from_documents(chunks, embeddings)
db.save_local(INDEX_DIR)

print(f"✅ Indexed {len(chunks)} chunks")
