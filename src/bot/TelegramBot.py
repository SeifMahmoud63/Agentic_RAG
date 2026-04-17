import logging
import httpx
import os
import io
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv


current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(os.path.dirname(current_dir), ".env")
load_dotenv(dotenv_path=env_path)

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
DEFAULT_PROJECT_ID = "telegram_uploads"

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "**RAG Assistant Bot Active**\n\n"
        "You can:\n"
        "**Ask a Question**: Just type it here!\n"
        "**Upload a Document**: Send a PDF or TXT file."
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.message.text
    if not query:
        return

    await update.message.reply_chat_action("typing")

    async with httpx.AsyncClient() as client:
        try:
            logger.info(f"Calling Ask_Q with query: {query}")
            response = await client.post(
                f"{API_BASE_URL}/api/v25/data/Ask_Q",
                json={"query": query},
                timeout=120.0
            )
            
            if response.status_code == 200:
                data = response.json()
                answer = data.get("answer", "No answer found.")
                try:
                    await update.message.reply_text(f"**Answer:**\n\n{answer}", parse_mode='Markdown')
                except Exception as parse_err:
                    logger.warning(f"Markdown parsing failed: {parse_err}. Sending as plain text.")
                    await update.message.reply_text(f"**Answer:**\n\n{answer}")
            else:
                logger.error(f"API Error ({response.status_code}): {response.text}")
                await update.message.reply_text(f"API Error ({response.status_code}): {response.text[:100]}")
        except Exception as e:
            logger.error(f"Exception in handle_message: {str(e)}")
            await update.message.reply_text(f"Error: {str(e)}")

async def handle_file_upload(update: Update, context: ContextTypes.DEFAULT_TYPE):
    document = update.message.document
    
    if not document:
        await update.message.reply_text("Please send a valid document (PDF/TXT).")
        return

    file_id = document.file_id
    file_name = document.file_name or "uploaded_file.pdf"

    await update.message.reply_text(f"Processing: {file_name}...")
    
    try:
        logger.info(f"Downloading file {file_id} from Telegram...")
        new_file = await context.bot.get_file(file_id)
        
        out = io.BytesIO()
        await new_file.download_to_memory(out)
        file_content = out.getvalue()
        
        if not file_content:
            await update.message.reply_text("Failed to download file from Telegram.")
            return

        await update.message.reply_chat_action("upload_document")

        async with httpx.AsyncClient() as client:
            logger.info(f"Uploading {file_name} to API...")
            
            import mimetypes
            mime_type, _ = mimetypes.guess_type(file_name)
            if not mime_type:
                mime_type = "application/pdf" if file_name.endswith(".pdf") else "text/plain"
                
            files = {'file': (file_name, bytes(file_content), mime_type)}
            upload_response = await client.post(
                f"{API_BASE_URL}/api/v25/data/upload/{DEFAULT_PROJECT_ID}",
                files=files,
                timeout=120.0
            )
            
            if upload_response.status_code != 200:
                logger.error(f"Upload failed: {upload_response.text}")
                await update.message.reply_text(f"API Upload Failed: {upload_response.text[:200]}")
                return

            upload_data = upload_response.json()
            internal_file_id = upload_data.get("file_id")
            
            # 3. Process the file
            await update.message.reply_text("Indexing into database...")
            logger.info(f"Processing file {internal_file_id}...")
            process_response = await client.post(
                f"{API_BASE_URL}/api/v25/data/process-assets",
                json={
                    "file_ids": [internal_file_id],
                    "project_id": DEFAULT_PROJECT_ID,
                    "chunk_size": 500,
                    "chunk_overlap": 50
                },
                timeout=120.0
            )

            if process_response.status_code == 200:
                process_data = process_response.json()
                status_signal = process_data.get("status")
                
                if status_signal in ["MADE_CHUNKS_SUCCESSFULY", "success"]:
                    await update.message.reply_text(f"Success! '{file_name}' is now searchable.")
                elif status_signal == "all_skipped":
                    await update.message.reply_text(f"'{file_name}' was skipped (identical content already exists).")
                else:
                    error_detail = process_data.get("detail", "Unknown error during processing.")
                    logger.error(f"Processing returned error status: {process_data}")
                    await update.message.reply_text(f"Indexing Failed! Reason: {error_detail}")
            else:
                logger.error(f"Processing failed: {process_response.text}")
                await update.message.reply_text(f"Uploaded, but indexing failed: {process_response.text[:200]}")

    except Exception as e:
        logger.error(f"Exception during upload: {str(e)}")
        await update.message.reply_text(f"System Error: {str(e)}")

def run_bot():
    if not TOKEN:
        print("CRITICAL: TELEGRAM_BOT_TOKEN not found in .env file.")
        return

    application = ApplicationBuilder().token(TOKEN).build()
    
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    
    application.add_handler(MessageHandler(filters.Document.ALL, handle_file_upload))
    
    print(f"--- Telegram Bot Starting (API: {API_BASE_URL}) ---")
    
    try:
        import httpx
        with httpx.Client() as client:
            response = client.get(f"{API_BASE_URL}/")
            if response.status_code == 200:
                print("API Backend is reachable.")
            else:
                print(f"Warning: API Backend returned status {response.status_code}")
    except Exception as e:
        print(f"Warning: Cannot reach API at {API_BASE_URL}. Is the FastAPI server running?")

    application.run_polling()

if __name__ == '__main__':
    run_bot()
