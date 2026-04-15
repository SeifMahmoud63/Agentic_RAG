cd src

python -m venv myenv

d:\RAG_MVC\src\myenv\Scripts\Activate.ps1

pip install -r requirements.txt



open docker 
docker run -d -p 6379:6379 redis

this is onther terminal and uvicorn in onther terminal

TTL cahse in bm25


uvicorn main:app --reload
python bot/telegram_bot.py


relevancy score / latency -> cache 75% and cache 0.1 (ttl = 1 H)