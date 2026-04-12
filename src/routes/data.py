from fastapi import FastAPI, APIRouter, Depends, UploadFile, status
from fastapi.responses import JSONResponse
import os
from helpers.config import get_settings, Settings
from controllers.DataController import DataController
from controllers.ProjectController import ProjectController
from controllers.ProcessController import ProcessController
from controllers.BaseController import BaseController

import aiofiles
from models import ResponseSignal

import logging
from .schema.data import ProcessRequest

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
    
    if not os.path.exists(base_path):
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"message": f"Path {base_path} not found!"}
        )

    for project_folder in os.listdir(base_path):
        project_path = os.path.join(base_path, project_folder)
        
        if os.path.isdir(project_path):
            controller = ProcessController(project_id=project_folder)
            
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
                    print(f"Skipping {project_folder}/{f_id}: {e}")
                    continue

    return {
        "status": "success",
        "total_chunks": len(all_chunks),
        "data": all_chunks
    }