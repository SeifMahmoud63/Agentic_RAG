"""
Ingestion Service — dedicated business logic layer.
Orchestrates: hash check → version logic → chunking → embedding → Qdrant upsert.
Separated from API routes for cleaner architecture, testability, and reuse by bot + API.
"""

import os
from logs.logger import logger
from typing import List, Optional, Dict, Any

from langchain_core.documents import Document

from helpers.hash_utils import generate_file_hash, generate_doc_hash
from helpers.config import get_settings
from controllers.process_controller import ProcessController
from vectordatabase import qdrant_db as QdrantDb, metadata_store as MetadataStore




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
    
    # --- STEP 1: Process Content First (to get chunk hashes for identification) ---
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
            "file_id": file_id,
            "detail": f"Chunking error: {str(e)}",
        }

    if not chunks:
        return {
            "status": "error",
            "file_id": file_id,
            "detail": "No chunks generated (file might be empty).",
        }

    # --- STEP 2: Identity Resolution (Who am I?) ---
    version = 1
    is_update = False
    use_file_id = file_id # Default to new ID
    match_source = "new"

    # A. First, check if the FILENAME already exists in this project
    existing_file = MetadataStore.get_file_by_name(clean_name, project_id)
    
    if existing_file:
        is_update = True
        use_file_id = existing_file["file_id"] 
        old_version = existing_file["current_version"]
        version = old_version + 1
        match_source = "filename"
        logger.info(f"UPDATE detected for '{clean_name}' (Filename Match). Anchor ID: {use_file_id}")
    else:
        # B. If no name match, perform GLOBAL CONTENT OVERLAP CHECK
        logger.info(f"Scanning project '{project_id}' for content overlap for '{clean_name}'...")
        new_chunk_hashes = [generate_doc_hash(c.page_content) for c in chunks]
        
        sim_file_id, overlap_pct = QdrantDb.find_file_by_content_overlap(
            chunk_hashes=new_chunk_hashes,
            project_id=project_id
        )

        if sim_file_id and overlap_pct >= settings.DUPLICATE_THRESHOLD:
            # We found a sibling! Adopt its identity.
            is_update = True
            use_file_id = sim_file_id
            
            # Get the version from the actual matched file
            sim_file_meta = MetadataStore.get_file_by_id(sim_file_id)
            old_name = sim_file_meta.get("original_name") if sim_file_meta else "Unknown"
            old_version = sim_file_meta.get("current_version", 1) if sim_file_meta else 1
            version = old_version + 1
            match_source = "content"
            
            logger.info(f"--- [IDENTITY MATCH] Detected {overlap_pct*100:.1f}% duplication with '{old_name}' (ID: {sim_file_id}). ---")
            logger.info(f"--- Treating '{clean_name}' as an update to '{old_name}'. ---")
        else:
            logger.info(f"No significant overlap found ({overlap_pct*100:.1f}% matched). Treating as NEW document.")

    # --- STEP 3: Execute Sync ---
    sync_summary = QdrantDb.sync_file_chunks(
        new_chunks=chunks,
        file_id=use_file_id,
        file_hash=file_hash,
        version=version,
        project_id=project_id,
    )

    # --- STEP 4: Update Metadata Registry ---
    if is_update:
        MetadataStore.register_new_version(use_file_id, file_hash)
        # Optional: If the name changed, update it in metadata to the new name? 
        # For now we keep the original anchor name, or we could update it.
    else:
        MetadataStore.register_new_file(
            file_id=use_file_id,
            original_name=clean_name,
            project_id=project_id,
            file_hash=file_hash,
        )

    action = "updated" if is_update else "ingested"
    logger.info(f"Successfully {action} '{clean_name}' via {match_source} logic: {sync_summary['indexed']} new, {sync_summary['kept']} kept (v{version}).")

    return {
        "status": "success",
        "action": action,
        "match_source": match_source,
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
