from .BaseController import BaseController
from fastapi import UploadFile
from models import ResponseEnum
from helpers import config


class DataController(BaseController):

    def __init__(self):
        super().__init__()
        self.size_sclae=1024*1024

    def validate_uploaded_file(self,file:UploadFile):
        # make it /uploadFile to can make content_type and check 


        if file.content_type not in self.app_settings.FILE_ALLOWED_TYPES:
            return False,ResponseEnum.FILE_TYPE_NOT_SUPPORTED.value
        
        if file.size > self.size_sclae * self.app_settings.FILE_MAX_SIZE:
            return False ,ResponseEnum.FILE_SIZE_NOT_EXCEEDED.value
        
        return True,ResponseEnum.FILE_VALIDATION_SUCC.value
        
       
    
# b2et data cont , b2et base cont , process