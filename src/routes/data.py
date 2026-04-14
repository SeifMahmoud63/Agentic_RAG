from fastapi import FastAPI, APIRouter, Depends, UploadFile, status
from fastapi.responses import JSONResponse
import os
from helpers.config import get_settings, Settings
from controllers.DataController import DataController
from controllers.ProjectController import ProjectController
from controllers.ProcessController import ProcessController
from controllers.BaseController import BaseController
from vectordatabase import database
from fastapi import HTTPException, status
import aiofiles
from models import ResponseSignal
import logging
from .schema.data import ProcessRequest
from vectordatabase import database
from retriever import retrieve_chunks
from Prompts import qa_prompt
from query_schema import QuestionRequest
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from Evaluation_RAGAS.evaluation import run_rag_evaluation
from helpers import hash_utils
from llm.llm import get_llm
from helpers import clean_response

logger = logging.getLogger('uvicorn.error')

data_router = APIRouter(
    prefix="/api/v25/data",
    tags=["api_v25", "data"],
)

@data_router.post("/upload/{project_id}")
async def upload_data(project_id: str, file: UploadFile,
                      app_settings: Settings = Depends(get_settings)):
        
    
    data_controller = DataController()

    is_valid, result_signal = data_controller.validate_uploaded_file(file=file)

    if not is_valid:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": result_signal
            }
        )

    project_dir_path = ProjectController().get_project_id(project_id=project_id)
    file_path, file_id = data_controller.generate_unique_filepath(
        orig_file_name=file.filename,
        project_id=project_id
    )

    try:
        async with aiofiles.open(file_path, "wb") as f:
            while chunk := await file.read(app_settings.FILE_DEFAULT_CHUNK_SIZE):
                await f.write(chunk)
    except Exception as e:

        logger.error(f"Error while uploading file: {e}")

        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": ResponseSignal.FILE_UPLOAD_FAILED.value
            }
        )

    return JSONResponse(
            content={
                "signal": ResponseSignal.FILE_VALIDATION_SUCC.value,
                "file_id": file_id
            }
        )

@data_router.post("/process-assets")
async def process_assets_folder(process_request: ProcessRequest):
    base_ctrl = BaseController()
    base_path = base_ctrl.file_dir
    all_chunks = []
    
    # If project_id is provided, only process that folder
    if process_request.project_id:
        target_folders = [process_request.project_id]
    else:
        # Otherwise, scan all folders
        if not os.path.exists(base_path):
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"message": f"Path {base_path} not found!"}
            )
        target_folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]

    for project_folder in target_folders:
        controller = ProcessController(project_id=project_folder)
        project_path = os.path.join(base_path, project_folder)
        
        if not os.path.exists(project_path):
            continue

        files_to_process = process_request.file_ids or os.listdir(project_path)
        
        for f_id in files_to_process:
            try:
                content = controller.get_content(file_id=f_id)
                if content:
                    chunks = controller.process(
                        file_content=content,
                        file_id=f_id,
                        chunk_size=process_request.chunk_size,
                        chunk_overlap=process_request.chunk_overlap
                    )
                    if chunks:
                        all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Skipping {project_folder}/{f_id}: {e}")
                continue

    if all_chunks:
        # Save to vector DB and invalidate cache (triggers multi-worker sync)
        result_signal, vector_store = database.vector_db(docs=all_chunks)
        
        # Small delay to ensure disk write is stable before workers read it
        import time
        time.sleep(0.5)
        
        from helpers.cache import invalidate_bm25_cache
        invalidate_bm25_cache()
        
        return {
            "status": ResponseSignal.MADE_CHUNKS_SUCCESSFULY.value,
            "message": result_signal,
            "total_chunks": len(all_chunks),
        }
    
    return {
        "status": ResponseSignal.ERROR_IN_MAKE_CHUNKS.value,
        "detail": "No chunks were generated (files might be missing or empty)"
    }


load_dotenv()
settings = get_settings()

llm = get_llm()

@data_router.post("/Ask_Q")
async def ask_question(query: QuestionRequest):

    try:
        v_store = database.vector_db()
        
        if v_store is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=ResponseSignal.ERROR_TO_FOUND_DATABASE_WHILE_ASKING.value
            )

        relevant_docs = retrieve_chunks.advanced_retrieve(vector_store=v_store, query=query.query)

        if not relevant_docs:
            return {
                "query": query.query,
                "answer": ResponseSignal.SORRY_TO_FIND_RELEVANT_DATA.value,
                "sources": []
            }

        context_text = "\n\n".join([doc.page_content for doc in relevant_docs])

        final_prompt = qa_prompt.format(context_text=context_text, query=query.query)

        response = llm.invoke(final_prompt) 
        cleaned_answer = clean_response.clean_llm_response(response.content)


        sources = []
        for doc in relevant_docs:
            clean_metadata = {}
            for key, value in doc.metadata.items():
                if hasattr(value, "item"): 
                    clean_metadata[key] = value.item()
                else:
                    clean_metadata[key] = value
            
            sources.append({
                "content": doc.page_content[:200] + "...", 
                "metadata": clean_metadata
            })

        return {
            "status": "success",
            "query": query.query, 
            "answer": cleaned_answer,
            "sources": sources
        }
    
    except Exception as e:
        print(f"Error in Ask_Q endpoint: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ResponseSignal.ERROR_WHILE_PROCESSING_QUESTION.value)
    

@data_router.post("/Evaluate_RAG")
async def evaluate_rag():
    try:
        v_store = database.vector_db()
        
        if v_store is None:
            raise HTTPException(status_code=404, detail="Database not found")

        df_results = run_rag_evaluation(v_store)
        df_results = df_results.fillna(0)


        summary = df_results[["faithfulness", "answer_relevancy"]].mean().to_dict()
        detailed = df_results.to_dict(orient="records")

        return {
            "status": "success",
            "overall_scores": summary,
            "detailed_results": detailed
        }

    except Exception as e:
        print(f"Eval Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))