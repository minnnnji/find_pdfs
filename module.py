import os
from typing import List, Tuple
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PIL import Image
import fitz  # PyMuPDF
import tempfile
import numpy as np
from dotenv import load_dotenv

load_dotenv()  # .env 파일의 환경변수를 불러옵니다

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
# DATA_DIR = "fine_pdfs/data"
DATA_DIR = "./"
VECTOR_DB_PATH = "vector_db"

def extract_text_and_images_from_pdf(pdf_path: str) -> Tuple[str, List[str]]:
    """PDF에서 텍스트와 이미지를 추출"""
    doc = fitz.open(pdf_path)
    text = ""
    image_paths = []
    os.makedirs("tmp", exist_ok=True)
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text += page.get_text()
        # 이미지 추출
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            ext = base_image["ext"]
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}", dir="tmp") as img_file:
                img_file.write(image_bytes)
                image_paths.append(img_file.name)
    return text, image_paths

def load_all_pdfs(data_dir: str) -> List[Tuple[str, str, List[str]]]:
    """data 폴더 내 모든 PDF 파일의 텍스트와 이미지 추출"""
    pdf_data = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_path = os.path.join(root, file)
                text, images = extract_text_and_images_from_pdf(pdf_path)
                pdf_data.append((pdf_path, text, images))
    return pdf_data

def create_documents(pdf_data: List[Tuple[str, str, List[str]]]) -> List[Document]:
    """PDF별로 Document 객체 생성 (텍스트만 임베딩)"""
    docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
    for pdf_path, text, images in pdf_data:
        for chunk in splitter.split_text(text):
            metadata = {
                "source": pdf_path,
                "images": images
            }
            docs.append(Document(page_content=chunk, metadata=metadata))
    return docs

def build_vector_db(docs: List[Document], persist_path: str):
    """문서 임베딩 후 벡터 DB에 저장"""
    embeddings = OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY)
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(persist_path)

def load_vector_db(persist_path: str):
    """벡터 DB 로드"""
    embeddings = OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY)
    return FAISS.load_local(persist_path, embeddings,  allow_dangerous_deserialization=True)

def search_query(query: str, db, k: int = 3):
    """
    검색어 임베딩 후 유사 문서 검색 (파일명 중복 없이 반환)
    유사도가 높은 이유(매칭된 chunk의 일부)도 함께 반환
    """
    # 쿼리 임베딩
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-3-small")
    query_emb = embeddings.embed_query(query)
    # FAISS에서 유사도와 함께 검색
    docs_and_scores = db.similarity_search_with_score(query, k=k)
    output = []
    seen_files = set()
    for doc, score in docs_and_scores:
        source = doc.metadata.get("source", "")
        file_name = os.path.basename(source)
        folder = os.path.dirname(source)
        if file_name not in seen_files:
            # 유사도가 높은 이유: 해당 chunk의 일부(앞부분 200자)와 점수
            matched_text = doc.page_content[:200].replace('\n', ' ')
            output.append({
                "file": file_name,
                "folder": folder,
                "matched_text": matched_text,
                "score": score
            })
            seen_files.add(file_name)
    return output

if __name__ == "__main__":
    # 1. PDF에서 텍스트/이미지 추출 및 임베딩 생성
    pdf_data = load_all_pdfs(DATA_DIR)
    print(pdf_data)
    docs = create_documents(pdf_data)
    build_vector_db(docs, VECTOR_DB_PATH)
    print("임베딩 및 벡터 DB 저장 완료.")

    # # 2. 검색어 입력 및 결과 출력
    db = load_vector_db(VECTOR_DB_PATH)
    query = input("검색어를 입력하세요: ")
    results = search_query(query, db)
    print("검색 결과:")
    for r in results:
        print(f"파일명: {r['file']}, 폴더 위치: {r['folder']}")
        print(f"유사도가 높게 나온 부분(일부): {r['matched_text']}")
        print(f"유사도 점수: {r['score']}")
        print("-" * 40)
