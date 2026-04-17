import hashlib


def generate_doc_hash(text: str, metadata: dict = None) -> str:
    """
We hash the text content only, because the metadata 
  (specifically the 'source') changes with every file upload due to the random prefix.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def generate_file_hash(file_path: str) -> str:
    """
    Compute SHA-256 hash of the entire file content (binary).
    Used for file-level deduplication.
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            sha256.update(block)
    return sha256.hexdigest()
