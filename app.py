import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from module import load_all_pdfs, create_documents, build_vector_db, load_vector_db, search_query, VECTOR_DB_PATH, DATA_DIR

st.set_page_config(layout="wide")
st.title("PDF 폴더 기반 검색 서비스")

# 1. 폴더 선택
st.sidebar.header("📁 폴더 선택")
# DATA_DIR 하위 폴더 목록 가져오기
def get_subfolders(data_dir):
    return [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]

folders = get_subfolders(DATA_DIR)
if not folders:
    st.warning("데이터 폴더에 하위 폴더가 없습니다. fine_pdfs/data/에 폴더를 추가하세요.")
    st.stop()

selected_folder = st.sidebar.selectbox("폴더를 선택하세요", folders)
selected_folder_path = os.path.join(DATA_DIR, selected_folder)

# 2. 선택한 폴더 내 PDF 파일 목록 표시
pdf_files = [f for f in os.listdir(selected_folder_path) if f.lower().endswith(".pdf")]
if not pdf_files:
    st.warning(f"선택한 폴더({selected_folder})에 PDF 파일이 없습니다.")
    st.stop()

st.subheader(f"선택한 폴더: {selected_folder}")
st.write("폴더 내 PDF 파일 목록:")
st.dataframe(pd.DataFrame({"파일명": pdf_files}), use_container_width=True)

# 3. 벡터 DB 준비 (최초 1회만 생성)
vector_db_path = os.path.join(VECTOR_DB_PATH, selected_folder)
if not os.path.exists(vector_db_path):
    with st.spinner("PDF 임베딩 및 벡터 DB를 생성 중입니다..."):
        pdf_data = load_all_pdfs(selected_folder_path)
        docs = create_documents(pdf_data)
        build_vector_db(docs, vector_db_path)
        st.success("임베딩 및 벡터 DB 생성 완료!")

# 4. 검색창 및 결과 표시
st.markdown("---")
st.subheader("🔍 PDF 내용 검색")
query = st.text_input("검색어를 입력하세요", "")

if query:
    db = load_vector_db(vector_db_path)
    results = search_query(query, db, k=10)
    if results:
        # 파일명, 폴더, 유사도가 높게 나온 부분, 유사도 점수 컬럼을 명확히 표에 표시
        df = pd.DataFrame(results)
        df = df.rename(columns={
            "file": "파일명",
            "folder": "폴더 위치",
            "matched_text": "유사도가 높게 나온 부분(일부)",
            "score": "유사도 점수"
        })
        st.dataframe(df, use_container_width=True)
        # 각 파일별로 유사도가 높게 나온 부분을 상세히 보고 싶을 때 expandable로 제공
        for idx, row in df.iterrows():
            with st.expander(f"파일명: {row['파일명']} - 유사도가 높게 나온 부분 자세히 보기"):
                st.write(row["유사도가 높게 나온 부분(일부)"])
                st.write(f"유사도 점수: {round(row['유사도 점수'], 2)}")
    else:
        st.info("검색 결과가 없습니다.")
