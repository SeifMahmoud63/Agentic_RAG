import os
from langchain_community.vectorstores import Chroma
from models import ResponseSignal
from embedding_model import emb_model
from helpers import config
from helpers.cache import invalidate_bm25_cache
from helpers.hash_utils import generate_doc_hash

def filter_duplicates(vectorstore, docs):
    existing = vectorstore.get()
    existing_ids = set(existing["ids"]) if existing and "ids" in existing else set()

    new_docs = []
    new_ids = []
    seen_in_batch = set() 

    for doc in docs:
        doc_id = generate_doc_hash(doc.page_content, doc.metadata)

        if doc_id not in existing_ids and doc_id not in seen_in_batch:  # ← CHECK BOTH
            doc.metadata["id"] = doc_id
            new_docs.append(doc)
            new_ids.append(doc_id)
            seen_in_batch.add(doc_id) 

    return new_docs, new_ids
def vector_db(docs=None, delete_ids=None, update_docs=None):
    embedding_model = emb_model.get_embedding()
    settings = config.get_settings()
    persist_dir = settings.persist_directory

    vectorstore = None
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embedding_model,
            collection_name=settings.collection_name
        )

    if docs:
        seen = set()
        unique_docs = []
        for doc in docs:
            doc_id = generate_doc_hash(doc.page_content, doc.metadata)
            if doc_id not in seen:        
                seen.add(doc_id)
                doc.metadata["id"] = doc_id
                unique_docs.append(doc)
        docs = unique_docs                

        if vectorstore is None:
            doc_ids = [d.metadata["id"] for d in docs]
            vectorstore = Chroma.from_documents(
                documents=docs,
                ids=doc_ids,
                embedding=embedding_model,
                collection_name=settings.collection_name,
                persist_directory=persist_dir
            )
            invalidate_bm25_cache()
            return ResponseSignal.VECTORDB_CREATED_SUCCESSFULLY.value, vectorstore

        else:
            new_docs, new_ids = filter_duplicates(vectorstore, docs)
            if not new_docs:
                return "No new documents to add (All duplicates)", vectorstore

            vectorstore.add_documents(documents=new_docs, ids=new_ids)
            invalidate_bm25_cache()
            return ResponseSignal.ADDED_IN_VECTORDB_SUCCESSFULLY.value, vectorstore

    return vectorstore
