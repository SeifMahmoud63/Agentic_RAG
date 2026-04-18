from fastapi import FastAPI, APIRouter, Depends, UploadFile, status
from fastapi.responses import JSONResponse
import os
from helpers.config import get_settings, Settings
from controllers.DataController import DataController
from controllers.ProjectController import ProjectController
from controllers.BaseController import BaseController
from fastapi import HTTPException, status
import aiofiles
from models import ResponseSignal
from logs.logger import logger
from .schema.data import ProcessRequest
from retriever import RetrieveChunks
from prompts import QaPrompt
from QuerySchema import QuestionRequest
from dotenv import load_dotenv
from EvaluationRagas.evaluation import run_rag_evaluation
from llm.llm import get_llm
from helpers import CleanResponse, redis
from VectorDatabase import IngestionService, QdrantDb
import time
from langchain_core.outputs import Generation
from langchain_core.messages import HumanMessage, AIMessage
from agent.graph import graph
from agent.graph import graph
import redis as redis_lib




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
        # Automatic cache reset on error
        try:
            cache = redis.get_cache()
            cache.clear()
        except:
            pass

        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": ResponseSignal.FILE_UPLOAD_FAILED.value
            }
        )

    try:
        cache = redis.get_cache()
        cache.clear()
    except Exception as e:
        logger.warning(f"Failed to clear cache during upload: {e}")

    return JSONResponse(
            content={
                "signal": ResponseSignal.FILE_VALIDATION_SUCC.value,
                "file_id": file_id,
                "original_name": file.filename,
            }
        )

@data_router.post("/process-assets")
async def process_assets_folder(process_request: ProcessRequest):
    base_ctrl = BaseController()
    base_path = base_ctrl.file_dir
    results = []
    
    # If project_id is provided, only process that folder
    if process_request.project_id:
        target_folders = [process_request.project_id]
    else:
        # Otherwise, scan all folders
        if not os.path.exists(base_path):
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"message": ResponseSignal.PATH_NOT_FOUND.value.format(base_path=base_path)}
            )
        target_folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]

    for project_folder in target_folders:
        project_path = os.path.join(base_path, project_folder)
        
        if not os.path.exists(project_path):
            continue

        files_to_process = process_request.file_ids or os.listdir(project_path)
        
        for f_id in files_to_process:
            try:
                result = IngestionService.ingest_file(
                    file_id=f_id,
                    original_name=f_id,
                    project_id=project_folder,
                    chunk_size=process_request.chunk_size,
                    chunk_overlap=process_request.chunk_overlap,
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Skipping {project_folder}/{f_id}: {e}")
                # Automatic cache reset on error
                try:
                    cache = redis.get_cache()
                    cache.clear()
                except:
                    pass
                
                results.append({
                    "status": ResponseSignal.STATUS_ERROR.value,
                    "file_id": f_id,
                    "detail": str(e),
                })
                continue

    # Summarize results
    successful = [r for r in results if r.get("status") == "success"]
    skipped = [r for r in results if r.get("status") == "skipped"]
    errors = [r for r in results if r.get("status") == "error"]

    # Always clear cache after asset processing to ensure the RAG sees new data
    try:
        cache = redis.get_cache()
        cache.clear()
    except Exception as e:
        logger.warning(f"Failed to clear cache after processing: {e}")

    total_chunks = sum(r.get("sync_details", {}).get("total", 0) for r in successful)

    if successful:
        return {
            "status": ResponseSignal.MADE_CHUNKS_SUCCESSFULY.value,
            "total_chunks": total_chunks,
            "ingested": len(successful),
            "skipped": len(skipped),
            "errors": len(errors),
            "details": results,
        }
    elif skipped and not errors:
        return {
            "status": ResponseSignal.STATUS_ALL_SKIPPED.value,
            "detail": ResponseSignal.ALL_SKIPPED_DUPLICATE.value,
            "details": results,
        }
    
    return {
        "status": ResponseSignal.ERROR_IN_MAKE_CHUNKS.value,
        "detail": ResponseSignal.NO_CHUNKS_GENERATED.value,
        "details": results,
    }


load_dotenv()
settings = get_settings()

llm = get_llm()

@data_router.post("/Ask_Q")
async def ask_question(query: QuestionRequest):
    try:
        overall_start = time.time()
        cache = redis.get_cache()
        clean_query = query.query.strip()
        
        try:
            cached_answer = cache.lookup(clean_query)
        except Exception as e:
            logger.warning(f"Manual cache lookup failed: {e}")
            cached_answer = None

        if cached_answer:
            return {
                "query": query.query,
                "answer": CleanResponse.clean_llm_response(cached_answer),
                "cache_status": ResponseSignal.CACHE_HIT.value
            }

        session_id = getattr(query, "project_id", "global_memory_session")
        config_run = {"configurable": {"thread_id": session_id}}
        input_state = {"messages": [HumanMessage(content=clean_query)]}
        
        final_state = await graph.ainvoke(input_state, config=config_run)

        agent_messages = [msg for msg in final_state["messages"] if isinstance(msg, AIMessage)]
        agent_response = agent_messages[-1].content if agent_messages else ResponseSignal.COULDNT_GENERATE_ANSWER.value
        cleaned_answer = CleanResponse.clean_llm_response(agent_response)

        try:
            cache.update(clean_query, cleaned_answer)
        except: pass

        print(f"=== TOTAL END-TO-END TIME: {time.time() - overall_start:.2f}s ===")
        return {
            "query": query.query,
            "answer": cleaned_answer,
            "cache_status": ResponseSignal.CACHE_MISS.value
        }
    except Exception as e:
        logger.error(f"Error in Ask_Q: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@data_router.post("/reset-cache")
def reset_cache():
    try:
        cache = redis.get_cache()

        
        r = redis_lib.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT)
        r.flushall()
        return {"status": ResponseSignal.STATUS_SUCCESS.value, "message": ResponseSignal.SEMANTIC_CACHE_CLEARED.value}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")


@data_router.post("/Evaluate_RAG")
def evaluate_rag():
    try:
        df_results = run_rag_evaluation()
        
        df_results = df_results.fillna(0)
        
        summary = {
            "faithfulness": df_results["faithfulness"].mean(),
            "answer_relevancy": df_results["answer_relevancy"].mean()
        }
        
        return {
            "status": ResponseSignal.STATUS_SUCCESS.value,
            "overall_scores": summary,
            "detailed_results": df_results.to_dict(orient="records")
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Eval Error: {str(e)}")
