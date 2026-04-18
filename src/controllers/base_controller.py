from helpers import config
import os
import string
import random

class BaseController:
    def __init__(self):

        self.app_settings=config.get_settings()

        self.base_dir=os.path.dirname(os.path.dirname(__file__)) # root

        self.file_dir=os.path.join(
            self.base_dir,
           "assets/files"
        )

    def generate_random_string(self, length: int=12):
        return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

        



