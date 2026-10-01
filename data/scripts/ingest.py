from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PDF_FOLDER = PROJECT_ROOT / "data" / "documents" / "pdf_files copy"
VECTOR_DB_PATH = PROJECT_ROOT / "vector_db" / "worldotutor_chroma_db"

if not PDF_FOLDER.exists():
    raise FileNotFoundError(f"PDF folder not found: {PDF_FOLDER}")

all_documents = []

for file_name in sorted(PDF_FOLDER.iterdir()):
    if file_name.is_file() and file_name.suffix.lower() == ".pdf":
        loader = PyPDFLoader(str(file_name))
        documents = loader.load()
        all_documents.extend(documents)
        print(f"Loaded: {file_name.name}")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(all_documents)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = Chroma(
    collection_name="worldotutor_documents",
    embedding_function=embeddings,
    persist_directory=str(VECTOR_DB_PATH)
)

vector_store.add_documents(chunks)

print(f"Total documents: {len(all_documents)}")
print(f"Total chunks: {len(chunks)}")
print("All PDFs successfully stored in ChromaDB!")