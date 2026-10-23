"""
Search routes — Text, Image, and Video search endpoints.
Video search uses Server-Sent Events for real-time progress streaming.
"""

import asyncio
import json
import logging
import os
import tempfile
import time
from typing import AsyncGenerator

import yt_dlp
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from google import genai
from google.genai import types

from dependencies import get_client
from schemas import (
    ImageSearchResponse,
    SearchResult,
    TextSearchRequest,
    TextSearchResponse,
    VideoSearchRequest,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/search", tags=["search"])


# ── Text Search ─────────────────────────────────────────────────

@router.post("/text", response_model=TextSearchResponse)
async def search_text(request: TextSearchRequest, client=Depends(get_client)):
    """Semantic text search against the indexed hadith database."""
    try:
        results = client.query(request.query, n_results=request.n_results)
        return TextSearchResponse(
            results=[SearchResult(**r) for r in results],
            query=request.query,
            total_results=len(results),
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Text search error: {e}")
        raise HTTPException(status_code=500, detail="Search failed. Please try again.")


# ---------------------------------------------------------
# IMAGE SEARCH ROUTE
# ---------------------------------------------------------
@router.post("/image")
async def search_by_image(file: UploadFile, n_results: int = Form(default=5), client=Depends(get_client)):
    try:
        image_bytes = await file.read()
        
        # Offload the Gemini API call to a background thread
        def process_image():
            response = client.gemini_client.models.generate_content(
                model='gemini-3.1-flash-lite',
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type=file.content_type),
                    "Extract any text present in the image (especially Urdu script). "
                    "Then, provide an accurate English translation of its semantic meaning. "
                    "Return the output as a clean JSON object with keys 'extracted_urdu' and 'english_translation'."
                ]
            )
            # Clean up the markdown block formatting
            raw_text = response.text.strip().replace('```json\n', '').replace('\n```', '')
            return json.loads(raw_text)

        parsed_data = await asyncio.to_thread(process_image)
        english_query = parsed_data.get("english_translation", "")
        
        # Vector DB Search
        results = await asyncio.to_thread(client.query, english_query, n_results)
        
        return {
            "extracted_urdu": parsed_data.get("extracted_urdu", ""),
            "extracted_query": english_query, 
            "results": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------------------------------------
# VIDEO SEARCH ROUTE (SSE STREAMING)
# ---------------------------------------------------------
@router.post("/video")
async def search_video(request: VideoSearchRequest, client=Depends(get_client)):
    
    async def event_generator():
        temp_dir = tempfile.mkdtemp()
        downloaded_file_path = None
        gemini_file = None
        
        try:
            # STEP 1: Download Audio via yt-dlp
            yield f"event: progress\ndata: {json.dumps({'step': 1, 'message': 'Extracting media from platform...'})}\n\n"
            
            def download_audio():
                ydl_opts = {
                    'format': 'bestaudio/best',
                    'outtmpl': os.path.join(temp_dir, 'extracted_audio.%(ext)s'),
                    'quiet': True,
                    'noplaylist': True
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(request.url, download=True)
                    return ydl.prepare_filename(info)
            
            downloaded_file_path = await asyncio.to_thread(download_audio)

            # STEP 2: Upload to Gemini Files API
            yield f"event: progress\ndata: {json.dumps({'step': 2, 'message': 'Uploading to inference engine...'})}\n\n"
            
            def upload_and_poll():
                uploaded = client.gemini_client.files.upload(file=downloaded_file_path)
                # Polling loop
                while True:
                    file_info = client.gemini_client.files.get(name=uploaded.name)
                    if file_info.state == "ACTIVE":
                        break
                    elif file_info.state == "FAILED":
                        raise Exception("Media processing failed on the server.")
                    time.sleep(2)
                return uploaded
            
            gemini_file = await asyncio.to_thread(upload_and_poll)

            # STEP 3: Generate Unified Transcript
            yield f"event: progress\ndata: {json.dumps({'step': 3, 'message': 'Extracting transcript and translating...'})}\n\n"
            
            def generate_translation():
                response = client.gemini_client.models.generate_content(
                    model='gemini-3.1-flash-lite',
                    contents=[
                        gemini_file,
                        "Analyze this video clip completely. "
                        "1. Listen to the audio track. If there is clear spoken language, transcribe and translate it into English. "
                        "2. Look at the visual frames. If there is written text on the screen, extract and translate it into English. "
                        "3. Consolidate the information from both the audio and the visuals into a single, cohesive English transcript. "
                        "Return ONLY the final plain English text. Do not include markdown formatting, timestamps, or conversational filler."
                    ]
                )
                return response.text.strip()
                
            transcript = await asyncio.to_thread(generate_translation)
            yield f"event: transcription\ndata: {json.dumps({'text': transcript})}\n\n"

            # STEP 4: Vector Search
            yield f"event: progress\ndata: {json.dumps({'step': 4, 'message': 'Querying vector database...'})}\n\n"
            
            search_results = await asyncio.to_thread(client.query, transcript, request.n_results)
            yield f"event: results\ndata: {json.dumps({'results': search_results, 'transcription': transcript, 'total_results': len(search_results)})}\n\n"

        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'detail': str(e)})}\n\n"
            
        finally:
            # STEP 5: Cleanup
            def cleanup():
                if gemini_file:
                    try:
                        client.gemini_client.files.delete(name=gemini_file.name)
                    except:
                        pass
                if downloaded_file_path and os.path.exists(downloaded_file_path):
                    try:
                        os.remove(downloaded_file_path)
                    except:
                        pass
                if os.path.exists(temp_dir):
                    try:
                        os.rmdir(temp_dir)
                    except:
                        pass
            
            await asyncio.to_thread(cleanup)

    return StreamingResponse(event_generator(), media_type="text/event-stream")
