import logging
import httpx
import os
import io
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv

# Load environment variables
# This ensures .env is found even if bit is started from the root directory
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(os.path.dirname(current_dir), ".env")
load_dotenv(dotenv_path=env_path)

# Configuration
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
DEFAULT_PROJECT_ID = "telegram_uploads"

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 **RAG Assistant Bot Active**\n\n"
        "You can:\n"
        "1️⃣ **Ask a Question**: Just type it here!\n"
        "2️⃣ **Upload a File**: Send a PDF, TXT, or Image (as document or photo)."
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
                    await update.message.reply_text(f"🤖 **Answer:**\n\n{answer}", parse_mode='Markdown')
                except Exception as parse_err:
                    logger.warning(f"Markdown parsing failed: {parse_err}. Sending as plain text.")
                    await update.message.reply_text(f"🤖 **Answer:**\n\n{answer}")
            else:
                logger.error(f"API Error ({response.status_code}): {response.text}")
                await update.message.reply_text(f"❌ API Error ({response.status_code}): {response.text[:100]}")
        except Exception as e:
            logger.error(f"Exception in handle_message: {str(e)}")
            await update.message.reply_text(f"❌ Error: {str(e)}")

async def handle_file_upload(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Detect if it's a document or a photo
    document = update.message.document
    photo = update.message.photo
    
    file_id = None
    file_name = "uploaded_file"
    
    if document:
        file_id = document.file_id
        file_name = document.file_name
    elif photo:
        # Take the largest photo size
        file_id = photo[-1].file_id
        file_name = f"photo_{file_id[:8]}.jpg"
    else:
        await update.message.reply_text("❌ No file detected.")
        return

    await update.message.reply_text(f"📥 Processing: {file_name}...")
    
    try:
        # 1. Download from Telegram
        logger.info(f"Downloading file {file_id} from Telegram...")
        new_file = await context.bot.get_file(file_id)
        
        # Modern python-telegram-bot (v20+) uses download_to_memory
        out = io.BytesIO()
        await new_file.download_to_memory(out)
        file_content = out.getvalue()
        
        if not file_content:
            await update.message.reply_text("❌ Failed to download file from Telegram.")
            return

        await update.message.reply_chat_action("upload_document")

        async with httpx.AsyncClient() as client:
            # 2. Upload to FastAPI
            logger.info(f"Uploading {file_name} to API...")
            files = {'file': (file_name, bytes(file_content))}
            upload_response = await client.post(
                f"{API_BASE_URL}/api/v25/data/upload/{DEFAULT_PROJECT_ID}",
                files=files,
                timeout=120.0
            )
            
            if upload_response.status_code != 200:
                logger.error(f"Upload failed: {upload_response.text}")
                await update.message.reply_text(f"❌ API Upload Failed: {upload_response.text[:200]}")
                return

            upload_data = upload_response.json()
            internal_file_id = upload_data.get("file_id")
            
            # 3. Process the file
            await update.message.reply_text("⚙️ Indexing into database...")
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
                await update.message.reply_text(f"✅ Success! '{file_name}' is now searchable.")
            else:
                logger.error(f"Processing failed: {process_response.text}")
                await update.message.reply_text(f"⚠️ Uploaded, but indexing failed: {process_response.text[:200]}")

    except Exception as e:
        logger.error(f"Exception during upload: {str(e)}")
        await update.message.reply_text(f"❌ System Error: {str(e)}")

def run_bot():
    if not TOKEN:
        print("CRITICAL: TELEGRAM_BOT_TOKEN not found in .env file.")
        return

    application = ApplicationBuilder().token(TOKEN).build()
    
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    
    # Combined handler for files and photos
    application.add_handler(MessageHandler(filters.Document.ALL | filters.PHOTO, handle_file_upload))
    
    print(f"--- Telegram Bot Starting (API: {API_BASE_URL}) ---")
    
    # Check if API is reachable
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
