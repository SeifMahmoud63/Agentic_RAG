"""
Ingestion Service — dedicated business logic layer.
Orchestrates: hash check → version logic → chunking → embedding → Qdrant upsert.
Separated from API routes for cleaner architecture, testability, and reuse by bot + API.
"""

import os
from logs.logger import logger
from typing import List, Optional, Dict, Any

from langchain_core.documents import Document

from helpers.HashUtils import generate_file_hash
from helpers.config import get_settings
from controllers.ProcessController import ProcessController
from VectorDatabase import QdrantDb, MetadataStore




def ingest_file(
    file_id: str,
    original_name: str,
    project_id: str,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Full ingestion pipeline for a single file.

    Steps:
    1. Compute SHA-256 hash of file content
    2. Check hash for exact duplicate → skip if identical content + same file group
    3. Check file name for update → delete old chunks + version up
    4. Chunk the file
    5. Embed + upsert to Qdrant
    6. Register in metadata store

    Returns a result dict with status and details.
    """
    settings = get_settings()
    if chunk_size is None:
        chunk_size = settings.chunk_size
    if chunk_overlap is None:
        chunk_overlap = settings.chunk_overlap

    controller = ProcessController(project_id=project_id)
    file_path = os.path.join(controller.project_path, file_id)

    if not os.path.exists(file_path):
        return {
            "status": "error",
            "file_id": file_id,
            "detail": f"File not found on disk: {file_path}",
        }

    file_hash = generate_file_hash(file_path)
    logger.info(f"File hash for '{file_id}': {file_hash[:16]}...")

    existing_by_hash = MetadataStore.hash_exists(file_hash)
    if existing_by_hash:
        existing_name = existing_by_hash.get("original_name", "")
        existing_project = existing_by_hash.get("project_id", "")

        if existing_project == project_id:
            logger.info(f"SKIP: Identical content already indexed as '{existing_name}' in project '{project_id}'.")
            return {
                "status": "skipped",
                "file_id": existing_by_hash["file_id"],
                "detail": f"Identical content already indexed (hash match). Original: {existing_name}",
                "file_hash": file_hash,
            }
        else:
            logger.info(f"Same content exists in project '{existing_project}', but allowing for project '{project_id}'.")

    clean_name = _extract_original_name(file_id)
    existing_file = MetadataStore.get_file_by_name(clean_name, project_id)

    version = 1
    is_update = False
    use_file_id = file_id # Default to new ID

    if existing_file:
        is_update = True
        use_file_id = existing_file["file_id"] 
        old_version = existing_file["current_version"]
        version = old_version + 1

        logger.info(f"UPDATE detected for '{clean_name}'. Syncing into existing anchor ID: {use_file_id}")
    else:
        logger.info(f"Indexing new document: {clean_name} (ID: {file_id})")

    try:
        content = controller.get_content(file_id=file_id)
        chunks = controller.process(
            file_content=content,
            file_id=file_id,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        logger.info(f"Chunking complete: Created {len(chunks)} chunks for '{clean_name}'.")
    except Exception as e:
        logger.error(f"Error chunking file '{file_id}': {e}")
        return {
            "status": "error",
            "file_id": use_file_id,
            "detail": f"Chunking error: {str(e)}",
        }

    if not chunks:
        return {
            "status": "error",
            "file_id": use_file_id,
            "detail": "No chunks generated (file might be empty).",
        }

    sync_summary = QdrantDb.sync_file_chunks(
        new_chunks=chunks,
        file_id=use_file_id,
        file_hash=file_hash,
        version=version,
    )

    if is_update:
        MetadataStore.register_new_version(use_file_id, file_hash)
    else:
        MetadataStore.register_new_file(
            file_id=use_file_id,
            original_name=clean_name,
            project_id=project_id,
            file_hash=file_hash,
        )

    action = "updated" if is_update else "ingested"
    logger.info(f"Successfully {action} '{clean_name}': {sync_summary['indexed']} new, {sync_summary['kept']} kept (v{version}).")

    return {
        "status": "success",
        "action": action,
        "file_id": use_file_id,
        "original_name": clean_name,
        "version": version,
        "sync_details": sync_summary,
        "file_hash": file_hash,
    }


def _extract_original_name(file_id: str) -> str:
    """
    Extract the original file name from a file_id like 'abc123def456_report.pdf'.
    The random prefix is 12 chars followed by an underscore.
    """
    if "_" in file_id:
        return file_id.split("_", 1)[1]
    return file_id
