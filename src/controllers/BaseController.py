from helpers import config
import os

class BaseController:
    def __init__(self):

        self.app_settings=config.get_settings()

        self.base_dir=os.path.dirname(os.path.dirname(__file__)) # root

        self.file_dir=os.path.join(
            self.base_dir,
           " assets/files"
        )

        



