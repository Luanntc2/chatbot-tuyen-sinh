import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from config import EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, FAISS_INDEX_PATH, DATA_DIR


def load_documents(data_dir: str):
    documents = []
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Thư mục dữ liệu không tồn tại: {data_dir}")

    txt_files = sorted(data_path.glob("*.txt"))
    pdf_files = sorted(data_path.glob("*.pdf"))

    for txt_file in txt_files:
        try:
            loader = TextLoader(str(txt_file), encoding="utf-8")
            docs = loader.load()
            documents.extend(docs)
            print(f"  [OK] {txt_file.name} — {len(docs)} document(s)")
        except Exception as e:
            print(f"  [SKIP] {txt_file.name} — lỗi: {e}")

    for pdf_file in pdf_files:
        try:
            loader = PyPDFLoader(str(pdf_file))
            docs = loader.load()
            documents.extend(docs)
            print(f"  [OK] {pdf_file.name} — {len(docs)} trang")
        except Exception as e:
            print(f"  [SKIP] {pdf_file.name} — lỗi: {e}")

    return documents, len(txt_files), len(pdf_files)


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    return chunks


def build_vectorstore(chunks):
    print("  Đang tải embedding model (lần đầu sẽ tải về từ HuggingFace)...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    print("  Đang tạo FAISS index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def ingest():
    print(f"=== Bắt đầu ingest dữ liệu từ '{DATA_DIR}' ===\n")

    print("1. Đọc tài liệu...")
    documents, n_txt, n_pdf = load_documents(DATA_DIR)
    total_files = n_txt + n_pdf

    if not documents:
        print("\n[LỖI] Không tìm thấy tài liệu nào. Hãy thêm file .txt hoặc .pdf vào thư mục data/raw/")
        return

    print(f"\n   Tổng: {total_files} file ({n_txt} .txt, {n_pdf} .pdf) → {len(documents)} document(s)\n")

    print("2. Chia chunk...")
    chunks = split_documents(documents)
    print(f"   Tạo ra {len(chunks)} chunk(s)\n")

    print("3. Tạo embeddings và lưu FAISS index...")
    vectorstore = build_vectorstore(chunks)

    os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)
    vectorstore.save_local(FAISS_INDEX_PATH)
    print(f"   Đã lưu vào '{FAISS_INDEX_PATH}'\n")

    print("=== Thống kê ===")
    print(f"  Số file đọc   : {total_files}")
    print(f"  Số chunk tạo  : {len(chunks)}")
    print(f"  FAISS index   : {FAISS_INDEX_PATH}")
    print("\nIngest hoàn thành!")


if __name__ == "__main__":
    ingest()
