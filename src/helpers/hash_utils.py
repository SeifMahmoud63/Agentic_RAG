import hashlib

def generate_doc_hash(text: str, metadata: dict = None) -> str:
    """
We hash the text content only, because the metadata 
  (specifically the 'source') changes with every file upload due to the random prefix.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()