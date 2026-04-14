import re

def clean_llm_response(text: str) -> str:

    cleaned_text = re.sub(r'\*+', '', text)
    cleaned_text = re.sub(r' +', ' ', cleaned_text).strip()
    
    return cleaned_text