import os
from langchain_community.vectorstores import Chroma
from models import ResponseSignal
from embedding_model import emb_model
from helpers import config

def vector_db(docs=None):
    embedding_model = emb_model.get_embedding()
    get_setting = config.get_settings()
    
    persist_dir = get_setting.persist_directory

    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embedding_model,
            collection_name=get_setting.collection_name
        )
        
        if docs:
            vectorstore.add_documents(docs)
            return ResponseSignal.ADDED_IN_VECTORDB_SUCCESSFULLY.value, vectorstore
        
        return vectorstore 

    else:
        if docs is None:
            return "Error: Docs required for initialization", None
            
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embedding_model,
            collection_name=get_setting.collection_name,
            persist_directory=persist_dir
        )
        return ResponseSignal.VECTORDB_CREATED_SUCCESSFULLY.value, vectorstore

# print("------------------------------------------------------------------------------------------------------------")

# print("el fo2 dh production ")

# print("------------------------------------------------------------------------------------------------------------")



# import os
# import shutil
# import gc
# from langchain_community.vectorstores import Chroma
# from langchain_community.document_loaders import PyPDFLoader, TextLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from models import ResponseSignal, ProcessingEnum
# from embedding_model import emb_model
# from helpers import config

# def vector_db():
#     embedding_model = emb_model.get_embedding()
#     get_setting = config.get_settings()
#     persist_dir = get_setting.persist_directory
    
#     # 1. تحديد مسار الفولدرات (1, 2, 3...)
#     assets_base_path = os.path.join(os.getcwd(), "assets", "files")

#     # 2. مسح الداتابيز القديمة عشان نضمن الـ Full Indexing النظيف
#     if os.path.exists(persist_dir):
#         gc.collect()
#         try:
#             shutil.rmtree(persist_dir)
#             print("--- Old VectorDB Deleted ---")
#         except Exception as e:
#             print(f"Warning: Could not delete old DB: {e}")

#     # 3. تجميع كل الملفات من كل الفولدرات الفرعية
#     all_docs = []
#     if os.path.exists(assets_base_path):
#         for root, dirs, files in os.walk(assets_base_path):
#             for file in files:
#                 file_path = os.path.join(root, file)
#                 file_ext = os.path.splitext(file).lower()[-1]
                
#                 loader = None
#                 if file_ext == ProcessingEnum.PDF.value:
#                     loader = PyPDFLoader(file_path)
#                 elif file_ext == ProcessingEnum.TXT.value:
#                     loader = TextLoader(file_path, encoding="utf-8")
                
#                 if loader:
#                     all_docs.extend(loader.load())
    
#     if not all_docs:
#         return "No documents found in assets/files", None

#     # 4. التقطيع (Chunking) لكل الداتا مع بعض
#     # تقدر تقرأ الـ chunk_size من الـ config لو حابب
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=config.get_setting()., 
#         chunk_overlap=150
#     )
#     all_chunks = text_splitter.split_documents(all_docs)

#     # 5. إنشاء الـ Vector Database الجديدة من الصفر
#     vectorstore = Chroma.from_documents(
#         documents=all_chunks,
#         embedding=embedding_model,
#         collection_name=get_setting.collection_name,
#         persist_directory=persist_dir
#     )

#     print(f"--- Full Indexing Complete: {len(all_chunks)} chunks created ---")
#     return ResponseSignal.VECTORDB_CREATED_SUCCESSFULLY.value, vectorstore