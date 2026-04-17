from enum import Enum

class ResponseSignal(Enum):

    FILE_TYPE_NOT_SUPPORTED="file type not SUPPORTED"
    FILE_SIZE_NOT_EXCEEDED="file size not EXCEEDED"
    FILE_VALIDATION_SUCC="file validation successfully"
    FILE_UPLOAD_FAILED="file upload failed"
    VECTORDB_CREATED_SUCCESSFULLY="VECTORDB_CREATED_SUCCESSFULLY"
    ADDED_IN_VECTORDB_SUCCESSFULLY="ADDED_IN_VECTORDB_SUCCESSFULLY"
    MADE_CHUNKS_SUCCESSFULY="MADE_CHUNKS_SUCCESSFULY"
    ERROR_IN_MAKE_CHUNKS="ERROR_IN_MAKE_CHUNKS"
    ERROR_TO_FOUND_DATABASE_WHILE_ASKING="Vector Database not found. Please process and index your assets first."
    SORRY_TO_FIND_RELEVANT_DATA="I'm sorry, I couldn't find any relevant information in the provided documents to answer your question.",
    ERROR_WHILE_PROCESSING_QUESTION="An internal error occurred while processing your question."
    FILE_ALREADY_INDEXED="File is already indexed (duplicate content detected)."
    FILE_VERSION_UPDATED="File version updated successfully."


