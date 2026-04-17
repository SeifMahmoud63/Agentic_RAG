import os

def get_prompt(file_name):
    current_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_dir, f"{file_name}.txt")
    
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

HYDE_PROMPT = get_prompt("hyde")
REWRITE_PROMPT = get_prompt("ReWriteQuery")
QaPrompt=get_prompt("QaPrompt")

# SystemPrompt has been moved to src/agent/SystemPrompt.txt
AGENT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "agent")
with open(os.path.join(AGENT_DIR, "SystemPrompt.txt"), "r", encoding="utf-8") as f:
    SystemPrompt = f.read()
