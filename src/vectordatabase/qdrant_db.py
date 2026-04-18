"""
Qdrant vector database wrapper.
Handles collection creation, upsert, filtered delete, and hybrid search.
Dense vectors: Google Gemini embeddings (3072-dim)
Sparse vectors: SPLADE via fastembed
"""

from logs.logger import logger
import uuid
from typing import List, Optional, Dict
from functools import lru_cache
from collections import defaultdict
import time
import random

from qdrant_client import QdrantClient, models
from fastembed import SparseTextEmbedding
from langchain_core.documents import Document

from helpers.config import get_settings
from helpers.hash_utils import generate_doc_hash
from embeddingmodel import emb_model as EmbModel
from models import ResponseSignal




@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    settings = get_settings()
    client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
    logger.info(f"Connected to Qdrant at {settings.QDRANT_HOST}:{settings.QDRANT_PORT}")
    return client


@lru_cache(maxsize=1)
def get_sparse_model() -> SparseTextEmbedding:
    """Load SPLADE sparse embedding model (cached singleton)."""
    logger.info("Loading SPLADE sparse embedding model...")
    model = SparseTextEmbedding(model_name=get_settings().SPLADEE_MODEL_NAME)
    logger.info("SPLADE model loaded.")
    return model


def warm_up():
    """Pre-load models at server startup to avoid first-request latency."""
    get_sparse_model()
    _ensure_collection()
    logger.info("Qdrant warm-up complete.")


DENSE_VECTOR_NAME = get_settings().QDRANT_DENSE_VECTOR_NAME
SPARSE_VECTOR_NAME = get_settings().QDRANT_SPARSE_VECTOR_NAME
DENSE_DIM = get_settings().QDRANT_DENSE_DIM


def _ensure_collection():
    """Create collection if it does not exist."""
    settings = get_settings()
    client = get_qdrant_client()
    collection_name = settings.QDRANT_COLLECTION

    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                DENSE_VECTOR_NAME: models.VectorParams(
                    size=DENSE_DIM,
                    distance=models.Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=False),
                ),
            },
        )
        client.create_payload_index(
            collection_name=collection_name,
            field_name="file_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        client.create_payload_index(
            collection_name=collection_name,
            field_name="chunk_hash",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        logger.info(f"Created Qdrant collection '{collection_name}' with payload indices on 'file_id' and 'chunk_hash'.")
    else:
        logger.info(f"Qdrant collection '{collection_name}' already exists.")



def upsert_chunks(
    chunks: List[Document],
    file_id: str,
    file_hash: str,
    version: int,
    project_id: str,
) -> int:
    """
    Embed and upsert document chunks into Qdrant.
    Returns the number of points upserted.
    """
    _ensure_collection()
    settings = get_settings()
    client = get_qdrant_client()
    collection_name = settings.QDRANT_COLLECTION

    dense_model = EmbModel.get_embedding()
    sparse_model = get_sparse_model()

    BATCH_SIZE = settings.QDRANT_UPSERT_BATCH_SIZE
    points = []
    
    for start in range(0, len(chunks), BATCH_SIZE):
        batch_chunks = chunks[start : start + BATCH_SIZE]
        batch_texts = [c.page_content for c in batch_chunks]

        batch_dense = None
        for attempt in range(settings.MAX_RETRIES):

            try:
                batch_dense = dense_model.embed_documents(batch_texts)
                break
            except Exception as e:
                err_msg = str(e).upper()
                if any(x in err_msg for x in ["429", "RESOURCE_EXHAUSTED", "RATE_LIMIT"]):
                    wait_time = 30 + random.uniform(5, 15)
                    logger.warning(f"⚠️ Quota Exceeded (429). Waiting {wait_time:.1f}s (Batch {start//BATCH_SIZE + 1})")
                    time.sleep(wait_time)
                else:
                    raise e
        
        if not batch_dense:
            raise Exception(ResponseSignal.EMBEDDING_BATCH_FAILED.value)


        batch_sparse = list(sparse_model.embed(batch_texts))

        for i, chunk in enumerate(batch_chunks):
            chunk_id = str(uuid.uuid4())
            chunk_text_hash = generate_doc_hash(chunk.page_content)
            
            sparse_vector = models.SparseVector(
                indices=batch_sparse[i].indices.tolist(),
                values=batch_sparse[i].values.tolist(),
            )

            payload = {
                "text": chunk.page_content,
                "file_id": file_id,
                "project_id": project_id,
                "chunk_id": chunk_id,
                "chunk_hash": chunk_text_hash,
                "file_hash": file_hash,
                "version": version,
            }
            if chunk.metadata:
                for k, v in chunk.metadata.items():
                    if k not in payload:
                        payload[k] = v.item() if hasattr(v, "item") else v

            point = models.PointStruct(
                id=chunk_id,
                vector={
                    DENSE_VECTOR_NAME: batch_dense[i],
                    SPARSE_VECTOR_NAME: sparse_vector,
                },
                payload=payload,
            )
            points.append(point)

    UPSERT_BATCH_SIZE = settings.QDRANT_POINTS_UPSERT_BATCH_SIZE
    for start in range(0, len(points), UPSERT_BATCH_SIZE):
        batch = points[start : start + UPSERT_BATCH_SIZE]
        client.upsert(collection_name=collection_name, points=batch)


    logger.info(f"Upserted {len(points)} chunks for file_id='{file_id}' (v{version}).")
    return len(points)


def find_file_by_content_overlap(
    chunk_hashes: List[str], 
    project_id: str
) -> tuple[Optional[str], float]:
    """
    Search globally across the project to find the file that has the highest 
    overlap with the provided chunk hashes.
    
    Returns: (best_file_id, overlap_percentage)
    """
    if not chunk_hashes:
        return None, 0.0

    _ensure_collection()
    client = get_qdrant_client()
    settings = get_settings()
    collection_name = settings.QDRANT_COLLECTION


    matches, _ = client.scroll(
        collection_name=collection_name,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(key="project_id", match=models.MatchValue(value=project_id)),
                models.FieldCondition(key="chunk_hash", match=models.MatchAny(any=chunk_hashes))
            ]
        ),
        with_payload=["file_id"],
        limit=settings.QDRANT_SCROLL_LIMIT_OVERLAP
    )

    if not matches:
        return None, 0.0

    file_counts = defaultdict(int)
    for p in matches:
        f_id = p.payload.get("file_id")
        if f_id:
            file_counts[f_id] += 1

    if not file_counts:
        return None, 0.0

    best_file_id = max(file_counts, key=file_counts.get)
    best_count = file_counts[best_file_id]
    

    overlap_pct = best_count / len(chunk_hashes)

    return best_file_id, overlap_pct


def sync_file_chunks(
    file_id: str,
    new_chunks: List[Document],
    file_hash: str,
    version: int,
    project_id: str,
) -> dict:
    """
    Incremental sync:
    1. Fetch existing chunk_hash values from Qdrant for this file.
    2. Identify chunks that already exist (skip).
    3. Identify chunks that are new or changed (upsert).
    4. Identify chunks that are missing in the new document (delete).
    """
    _ensure_collection()
    client = get_qdrant_client()
    settings = get_settings()
    collection_name = settings.QDRANT_COLLECTION


    existing_map = defaultdict(list)
    offset = None
    
    while True:
        existing_points, offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="file_id", match=models.MatchValue(value=file_id))]
            ),
            with_payload=["chunk_hash"],
            limit=settings.QDRANT_SCROLL_LIMIT_SYNC, 
            offset=offset
        )
        
        for p in existing_points:
            if "chunk_hash" in p.payload:
                existing_map[p.payload["chunk_hash"]].append(p.id)
        
        if offset is None:
            break
    
   
    chunks_to_upsert = []
    hashes_staying = set()
    
    for chunk in new_chunks:
        h = generate_doc_hash(chunk.page_content)
        if h in existing_map:
            hashes_staying.add(h)
        else:
            chunks_to_upsert.append(chunk)

    ids_to_delete = []
    for h, p_ids in existing_map.items():
        if h not in hashes_staying:
            ids_to_delete.extend(p_ids)
    
    num_indexed = 0
    if chunks_to_upsert:
        num_indexed = upsert_chunks(
            chunks=chunks_to_upsert, 
            file_id=file_id, 
            file_hash=file_hash, 
            version=version,
            project_id=project_id
        )
        
    if ids_to_delete:
        client.delete(
            collection_name=collection_name,
            points_selector=models.PointIdsList(points=ids_to_delete)
        )
        logger.info(f"Deleted {len(ids_to_delete)} orphaned chunks for '{file_id}'.")

    summary = {
        "indexed": num_indexed,
        "kept": len(hashes_staying),
        "deleted": len(ids_to_delete),
        "total": len(new_chunks)
    }
    logger.info(f"Sync complete for '{file_id}': {summary['kept']} kept, {summary['indexed']} added, {summary['deleted']} removed.")
    return summary



def delete_by_file_id(file_id: str) -> int:
    """
    Delete all points matching the given file_id using a filtered delete.
    Returns approximate count of deleted points.
    """
    _ensure_collection()
    
    settings = get_settings()
    client = get_qdrant_client()
    collection_name = settings.QDRANT_COLLECTION

    count_result = client.count(
        collection_name=collection_name,
        count_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="file_id",
                    match=models.MatchValue(value=file_id),
                )
            ]
        ),
    )
    deleted_count = count_result.count

    if deleted_count > 0:
        client.delete(
            collection_name=collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="file_id",
                            match=models.MatchValue(value=file_id),
                        )
                    ]
                )
            ),
        )
        logger.info(f"Deleted {deleted_count} chunks for file_id='{file_id}'.")
    else:
        logger.info(f"No chunks found for file_id='{file_id}' to delete.")

    return deleted_count




def hybrid_search(query: str, top_k: Optional[int] = None, dense_query: Optional[str] = None) -> List[Document]:
    """
    Perform hybrid search: dense (semantic) + sparse (keyword) with RRF fusion.
    Returns list of LangChain Document objects.
    """
    _ensure_collection()
    settings = get_settings()
    client = get_qdrant_client()
    collection_name = settings.QDRANT_COLLECTION

    if top_k is None:
        top_k = settings.TOP_K_HYBRID

    dense_weight = settings.DENSE_SEARCH_WEIGHT
    sparse_weight = settings.SPARSE_SEARCH_WEIGHT

    dense_model = EmbModel.get_embedding()
    sparse_model = get_sparse_model()

    query_dense = dense_model.embed_query(dense_query or query)
    query_sparse_raw = list(sparse_model.embed([query]))[0]
    query_sparse = models.SparseVector(
        indices=query_sparse_raw.indices.tolist(),
        values=query_sparse_raw.values.tolist(),
    )

    prefetch_limit = top_k

    results = client.query_points(
        collection_name=collection_name,
        prefetch=[
            models.Prefetch(
                query=query_dense,
                using=DENSE_VECTOR_NAME,
                limit=prefetch_limit,
            ),
            models.Prefetch(
                query=query_sparse,
                using=SPARSE_VECTOR_NAME,
                limit=prefetch_limit,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=top_k,
    )

    documents = []
    for point in results.points:
        payload = point.payload or {}
        text = payload.pop("text", "")
        doc = Document(page_content=text, metadata=payload)
        documents.append(doc)

    logger.info(f"Hybrid search returned {len(documents)} results for query: '{query[:50]}...'")
    return documents
