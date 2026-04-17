from .BaseController import BaseController
import os
from .ProjectController import ProjectController
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from models import ProcessingEnum
from langchain_text_splitters import RecursiveCharacterTextSplitter
from helpers import config


class ProcessController(BaseController):

    def __init__(self,project_id:str):
        super().__init__()
        self.project_id=project_id
        self.project_path=ProjectController().get_project_id(project_id=project_id)

    def get_file_extension(self,file_id:str):
            return os.path.splitext(file_id)[-1].lower()
    
   
    def get_loader(self,file_id:str):
            file_ext=self.get_file_extension(file_id=file_id)
            file_path=os.path.join(
                self.project_path,
                file_id
            )

            if file_ext == ProcessingEnum.TXT.value:
                return TextLoader(file_path,encoding="utf-8")
            
            if file_ext == ProcessingEnum.PDF.value:
                return PyPDFLoader(file_path)
            
            return None
        
    def get_content(self,file_id:str):
            loader=self.get_loader(file_id=file_id)
            return loader.load()
            

    def process(self,file_content:list,file_id:str,chunk_size:int=config.get_settings().chunk_size,chunk_overlap:int=config.get_settings().chunk_overlap):
            text_splitter=RecursiveCharacterTextSplitter(
                chunk_overlap=chunk_overlap,
                chunk_size=chunk_size
            )

            chunks = text_splitter.split_documents(file_content)

            return chunks

