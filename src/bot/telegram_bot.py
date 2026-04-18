import logging
import httpx
import os
import io
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv
import httpx


from helpers.config import get_settings
from models import ResponseSignal
from logs.logger import logger

settings = get_settings()
TOKEN = settings.TELEGRAM_BOT_TOKEN
API_BASE_URL = settings.API_BASE_URL
DEFAULT_PROJECT_ID = settings.TG_DEFAULT_PROJECT_ID

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(ResponseSignal.TG_BOT_ACTIVE_MSG.value)

async def process_query(update: Update, context: ContextTypes.DEFAULT_TYPE, query: str):
    await update.message.reply_chat_action("typing")

    async with httpx.AsyncClient() as client:
        try:
            logger.info(f"Calling Ask_Q with query: {query}")
            response = await client.post(
                f"{API_BASE_URL}/api/v25/data/Ask_Q",
                json={"query": query},
                timeout=settings.TG_REQUEST_TIMEOUT
            )
            
            if response.status_code == 200:
                data = response.json()
                answer = data.get("answer", ResponseSignal.TG_NO_ANSWER_FOUND.value)
                try:
                    await update.message.reply_text(f"**Answer:**\n\n{answer}", parse_mode='Markdown')
                except Exception as parse_err:
                    logger.warning(f"Markdown parsing failed: {parse_err}. Sending as plain text.")
                    await update.message.reply_text(f"**Answer:**\n\n{answer}")
            else:
                logger.error(f"API Error ({response.status_code}): {response.text}")
                await update.message.reply_text("error")
        except Exception as e:
            logger.error(f"Exception in process_query: {str(e)}")
            await update.message.reply_text("error")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.message.text
    if not query:
        return
    await process_query(update, context, query)

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    voice = update.message.voice
    if not voice:
        return

    try:
        logger.info(f"Downloading voice message {voice.file_id}...")
        new_file = await context.bot.get_file(voice.file_id)
        
        out = io.BytesIO()
        await new_file.download_to_memory(out)
        audio_bytes = out.getvalue()
        
        if not audio_bytes:
            await update.message.reply_text(ResponseSignal.TG_FAILED_DOWNLOAD_VOICE.value)
            return

        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            await update.message.reply_text(ResponseSignal.TG_STT_KEY_MISSING.value)
            return
            
        await update.message.reply_text(ResponseSignal.TG_TRANSCRIBING_VOICE.value)
        
        async with httpx.AsyncClient() as client:
            files = {'file': ('voice.ogg', audio_bytes, 'audio/ogg')}
            data = {
                'model': settings.TG_WHISPER_MODEL,
                'language': settings.TG_WHISPER_LANGUAGE
            }
            headers = {'Authorization': f'Bearer {settings.GROQ_API_KEY}'}
            
            logger.info("Sending audio to Groq Whisper API...")
            transcription_response = await client.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                files=files,
                data=data,
                headers=headers,
                timeout=settings.TG_REQUEST_TIMEOUT
            )
            
            if transcription_response.status_code != 200:
                logger.error(f"Transcription failed: {transcription_response.text}")
                await update.message.reply_text("error")
                return
                
            transcription_data = transcription_response.json()
            query = transcription_data.get("text", "").strip()
            
            if not query:
                await update.message.reply_text(ResponseSignal.TG_NO_CLEAR_SPEECH.value)
                return
            
            await update.message.reply_text(f"🗣️ Heard: {query}")
            
            await process_query(update, context, query)
            
    except Exception as e:
        logger.error(f"Exception during voice processing: {str(e)}")
        await update.message.reply_text("error")

async def handle_file_upload(update: Update, context: ContextTypes.DEFAULT_TYPE):
    document = update.message.document
    
    if not document:
        await update.message.reply_text(ResponseSignal.TG_SEND_VALID_DOC.value)
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
            await update.message.reply_text(ResponseSignal.TG_FAILED_DOWNLOAD_FILE.value)
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
                timeout=settings.TG_REQUEST_TIMEOUT
            )
            
            if upload_response.status_code != 200:
                logger.error(f"Upload failed: {upload_response.text}")
                await update.message.reply_text("error")
                return

            upload_data = upload_response.json()
            internal_file_id = upload_data.get("file_id")
            
            await update.message.reply_text(ResponseSignal.TG_INDEXING_DB.value)
            logger.info(f"Processing file {internal_file_id}...")
            process_response = await client.post(
                f"{API_BASE_URL}/api/v25/data/process-assets",
                json={
                    "file_ids": [internal_file_id],
                    "project_id": DEFAULT_PROJECT_ID,
                    "chunk_size": settings.TG_CHUNK_SIZE,
                    "chunk_overlap": settings.TG_CHUNK_OVERLAP
                },
                timeout=settings.TG_REQUEST_TIMEOUT
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
                await update.message.reply_text("error")

    except Exception as e:
        logger.error(f"Exception during upload: {str(e)}")
        await update.message.reply_text("error")

def run_bot():
    if not TOKEN:
        print("CRITICAL: TELEGRAM_BOT_TOKEN not found in .env file.")
        return

    application = ApplicationBuilder().token(TOKEN).build()
    
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    application.add_handler(MessageHandler(filters.VOICE, handle_voice))
    application.add_handler(MessageHandler(filters.Document.ALL, handle_file_upload))
    
    print(f"--- Telegram Bot Starting (API: {API_BASE_URL}) ---")
    
    try:
        
        with httpx.Client() as client:
            response = client.get(f"{API_BASE_URL}/api/v1/")
            if response.status_code == 200:
                print("API Backend is reachable.")
            else:
                print(f"Warning: API Backend returned status {response.status_code}")
    except Exception as e:
        print(f"Warning: Cannot reach API at {API_BASE_URL}. Is the FastAPI server running?")

    application.run_polling()

if __name__ == '__main__':
    run_bot()
