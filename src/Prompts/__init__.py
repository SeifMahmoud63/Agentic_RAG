import os

def get_prompt(file_name):
    current_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_dir, f"{file_name}.txt")
    
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

HYDE_PROMPT = get_prompt("hyde")
REWRITE_PROMPT = get_prompt("rewrite_query")
QA_PROMPT = get_prompt("qa_prompt")
