# from langchain_community.vectorstores import Chroma
# from models import ResponseSignal
# from embedding_model import emb_model
# import os
# from helpers import config
# from models import ResponseSignal

# def vector_db(docs=None):

#     embedding_model = emb_model.get_embedding()
#     get_setting=config.get_settings()

#     if os.path.exists() and os.listdir(get_setting.persist_directory):

#         vectorstore = Chroma(
#             persist_directory=get_setting.persist_directory,
#             embedding_function=embedding_model,
#             collection_name=get_setting.collection_name
#         )
        
#         if docs:
#             vectorstore.add_documents(docs)
#             return ResponseSignal.ADDED_IN_VECTORDB_SUCCESSFULLY.value
    
#     else:
#         if docs is None:
#             raise ValueError("Docs required for first initialization of the Vector DB")
            
#         vectorstore = Chroma.from_documents(
#             documents=docs,
#             embedding=embedding_model,
#             collection_name=get_setting.collection_name,
#             persist_directory=get_setting.persist_directory
#         )
#         return ResponseSignal.VECTORDB_CREATED_SUCCESSFULLY.value

#     return vectorstore


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