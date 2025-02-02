import chromadb
from typing import List, Dict

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import logging
from typing import List, Dict, Optional
import os
import hashlib
import torch
from PyPDF2 import PdfReader
import re

import yt_dlp
import moviepy
from moviepy import AudioFileClip
import whisper
import numpy as np

import os
from groq import Groq
from PIL import Image

from groq import Groq
import base64
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IslamicSourceFinderClient:
    _instance = None
    _is_initialized = False
    _model = None
    
    def __new__(cls, persist_directory: str = "/projects/ISF/chromadb_data"):
        if cls._instance is None:
            cls._instance = super(IslamicSourceFinderClient, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, persist_directory: str = "/projects/ISF/chromadb_data"):
        if not self._is_initialized:
            self.persist_directory = persist_directory
            self.client = self._initialize_chromadb()
            self.collection_name = "islamic_scripture"
            self.chunk_size = 1000
            self.processed_files = set()
            self._is_initialized = True
            
            # Create or load the collection
            self.collection = self._create_collection()
    
    @classmethod
    def get_model(cls):
        """Lazy load the model only when needed"""
        if cls._model is None:
            logger.info("Initializing SentenceTransformer model...")
            cls._model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        return cls._model
            
    def _initialize_chromadb(self) -> chromadb.Client:
        """Initialize ChromaDB client with error handling."""
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            client = chromadb.PersistentClient(path=self.persist_directory)
            logger.info("ChromaDB client initialized successfully")
            return client
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {str(e)}")
            raise

    def _create_collection(self) -> chromadb.Collection:
        """Create or get collection with error handling."""
        try:
            return self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            logger.error(f"Failed to create/get collection: {str(e)}")
            raise

    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate MD5 hash of file content."""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as file:
            buf = file.read(65536)  # Read in 64kb chunks
            while len(buf) > 0:
                hasher.update(buf)
                buf = file.read(65536)
        return hasher.hexdigest()

    def _is_file_processed(self, file_path: str) -> bool:
        """Check if file has already been processed using metadata."""
        file_hash = self._calculate_file_hash(file_path)
        try:
            # Check if any document exists with this file hash
            existing_docs = self.collection.get(
                where={"file_hash": file_hash}
            )
            return len(existing_docs['ids']) > 0
        except Exception:
            return False

    @staticmethod
    def _batch_generator(items, batch_size=32):
        """Generate batches of items"""
        for i in range(0, len(items), batch_size):
            yield items[i:i + batch_size]

    # def process_text_file(self, file_path: str, force_reload: bool = False) -> None:
    #     """Process text file and store in ChromaDB if not already processed."""
    #     try:
    #         if not force_reload and self._is_file_processed(file_path):
    #             logger.info(f"File {file_path} has already been processed. Skipping...")
    #             return

    #         # Calculate file hash
    #         file_hash = self._calculate_file_hash(file_path)

    #         # Read and process the file
    #         with open(file_path, 'r', encoding='utf-8') as file:
    #             text = file.read()

    #         # Chunk the text
    #         chunks = self._chunk_text(text)
            
    #         # Process in batches to reduce memory usage
    #         batch_size = 32
    #         all_embeddings = []
            
    #         for batch in self._batch_generator(chunks, batch_size):
    #             # Generate embeddings using lazy-loaded model
    #             with torch.no_grad():  # Disable gradient calculation for inference
    #                 batch_embeddings = self.get_model().encode(batch)
    #                 all_embeddings.extend(batch_embeddings.tolist())
            
    #         # Add documents to collection with file hash in metadata
    #         self.collection.add(
    #             documents=chunks,
    #             embeddings=all_embeddings,
    #             metadatas=[{
    #                 "source": file_path,
    #                 "chunk_id": i,
    #                 "file_hash": file_hash
    #             } for i in range(len(chunks))],
    #             ids=[f"chunk_{file_hash}_{i}" for i in range(len(chunks))]
    #         )
            
    #         logger.info(f"Successfully processed and stored {len(chunks)} chunks from {file_path}")
            
    #     except Exception as e:
    #         logger.error(f"Error processing file {file_path}: {str(e)}")
    #         raise
    

    # def process_text_file(self, file_path: str, force_reload: bool = False) -> None:
    #     try:
    #         if not force_reload and self._is_file_processed(file_path):
    #             logger.info(f"File {file_path} has already been processed. Skipping...")
    #             return

    #         # Calculate file hash
    #         file_hash = self._calculate_file_hash(file_path)

    #         # Extract text based on file type
    #         if file_path.lower().endswith(".pdf"):
    #             reader = PdfReader(file_path)
    #             text = ""
    #             for page in reader.pages:
    #                 text += page.extract_text()
    #         else:
    #             with open(file_path, 'r', encoding='utf-8') as file:
    #                 text = file.read()

    #         # Extract chunks with metadata
    #         chunks_with_metadata = self._chunk_text(text)

    #         # Prepare for storage
    #         documents = [chunk["chunk_text"] for chunk in chunks_with_metadata]
    #         metadatas = [{
    #             "source": file_path,
    #             "book_number": chunk["book_number"],
    #             "hadees_number": chunk["hadees_number"],
    #             "file_hash": file_hash
    #         } for chunk in chunks_with_metadata]
    #         ids = [f"chunk_{file_hash}_{i}" for i in range(len(chunks_with_metadata))]

    #         # Generate embeddings and add to collection
    #         batch_size = 32
    #         all_embeddings = []
    #         for batch in self._batch_generator(documents, batch_size):
    #             with torch.no_grad():
    #                 batch_embeddings = self.get_model().encode(batch)
    #                 all_embeddings.extend(batch_embeddings.tolist())

    #         self.collection.add(
    #             documents=documents,
    #             embeddings=all_embeddings,
    #             metadatas=metadatas,
    #             ids=ids
    #         )

    #         logger.info(f"Successfully processed and stored {len(documents)} chunks from {file_path}")

    #     except Exception as e:
    #         logger.error(f"Error processing file {file_path}: {str(e)}")
    #         raise
    def process_text_file(self, file_path: str, force_reload: bool = True) -> None:
        try:
            if not force_reload and self._is_file_processed(file_path):
                logger.info(f"File {file_path} has already been processed. Skipping...")
                return

            # Calculate file hash
            file_hash = self._calculate_file_hash(file_path)
            
            # Get the PDF filename without path
            pdf_name = os.path.basename(file_path)

            # Extract text based on file type
            if file_path.lower().endswith(".pdf"):
                reader = PdfReader(file_path)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
            else:
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()

            # Extract chunks with metadata
            chunks_with_metadata = self._chunk_text(text)

            # Prepare for storage
            documents = [chunk["chunk_text"] for chunk in chunks_with_metadata]
            metadatas = [{
                "source": file_path,
                "pdf_name": pdf_name,  # Add PDF name to metadata
                "book_number": chunk["book_number"],
                "hadees_number": chunk["hadees_number"],
                "file_hash": file_hash
            } for chunk in chunks_with_metadata]
            
            ids = [f"chunk_{file_hash}_{i}" for i in range(len(chunks_with_metadata))]

            # Generate embeddings and add to collection
            batch_size = 32
            all_embeddings = []
            for batch in self._batch_generator(documents, batch_size):
                with torch.no_grad():
                    batch_embeddings = self.get_model().encode(batch)
                    all_embeddings.extend(batch_embeddings.tolist())

            self.collection.add(
                documents=documents,
                embeddings=all_embeddings,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Successfully processed and stored {len(documents)} chunks from {file_path}")
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            raise

    def query(self, query_text: str, n_results: int = 3) -> List[Dict[str, str]]:
        """
        Query the collection and format the results to include PDF name, Book # and Hadees #.
        """
        try:
            with torch.no_grad():
                query_embedding = self.get_model().encode(query_text)

            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )

            # Ensure the response structure is valid
            if not results.get('documents') or not results['documents'][0]:
                raise ValueError("No results found for the query.")

            # Structure results with PDF name, Book # and Hadees #
            structured_results = []
            for i in range(len(results['documents'][0])):
                metadata = results['metadatas'][0][i]
                result_text = (
                    f"Book {metadata.get('book_number', 'Unknown')}, "
                    f"Hadees {metadata.get('hadees_number', 'Unknown')} "
                    f"(Source: {metadata.get('pdf_name', 'Unknown')}) "
                    f"(Relevance: {1 - results['distances'][0][i]:.2f})\n\n"
                    f"{results['documents'][0][i]}"
                )
                structured_results.append({
                    'pdf_name': metadata.get("pdf_name", "Unknown"),
                    'book_number': metadata.get("book_number", "Unknown"),
                    'hadees_number': metadata.get("hadees_number", "Unknown"),
                    'text': result_text,
                    'relevance_score': 1 - results['distances'][0][i]
                })

            return structured_results
        except Exception as e:
            logger.error(f"Error querying collection: {str(e)}")
            raise

    def process_multiple_pdfs(self):
        """
        Process multiple PDF files into the database.
        
        Args:
            pdf_paths (list): List of file paths to PDF documents
        """
        pdf_paths = ["muslim.pdf", "bukhari.pdf", "tirmidhi_vol1.pdf"]
        # Get the directory where the PDFs are stored
        pdf_dir = os.path.join(os.path.dirname(__file__), "data")  # Assuming PDFs are in a 'data' subdirectory
        
        for pdf_name in pdf_paths:
            try:
                # Construct full path
                pdf_path = os.path.join(pdf_dir, pdf_name)
                
                # Check if file exists
                if not os.path.exists(pdf_path):
                    logger.error(f"PDF file not found: {pdf_path}")
                    continue
                    
                logger.info(f"Starting to process: {pdf_path}")
                
                # Force reload to ensure new content is processed
                self.process_text_file(pdf_path, force_reload=True)
                
                logger.info(f"Successfully processed {pdf_path}")
                
            except Exception as e:
                logger.error(f"Error processing {pdf_path}: {str(e)}")
                logger.exception("Full error trace:")

    def _chunk_text(self, text: str) -> List[Dict[str, str]]:
        """
        Split text into chunks while extracting Book # and Hadees # for metadata.
        """
        chunks = []
        pattern = r"Book (\d+), Number (\d+):"  # Example regex for matching Book and Hadees #
        matches = list(re.finditer(pattern, text))

        for i, match in enumerate(matches):
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            chunk_text = text[start:end].strip()

            chunks.append({
                "book_number": match.group(1),
                "hadees_number": match.group(2),
                "chunk_text": chunk_text
            })

        return chunks

    def encode_image(self,image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
        
    def groq_ocr(self,image_path):
        

        # Getting the base64 string
        base64_image = self.encode_image(image_path)

        client = Groq()

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "provide only the text in the image"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ],
            model="llama-3.2-11b-vision-preview",
        )
        print(chat_completion.choices[0].message.content)
        return chat_completion.choices[0].message.content
    
    # def query(self, query_text: str, n_results: int = 3) -> List[Dict[str, str]]:
    #     """
    #     Query the collection and format the results to include Book # and Hadees #.
    #     """
    #     try:
    #         with torch.no_grad():
    #             query_embedding = self.get_model().encode(query_text)

    #         results = self.collection.query(
    #             query_embeddings=[query_embedding.tolist()],
    #             n_results=n_results,
    #             include=["documents", "metadatas", "distances"]
    #         )

    #         # Ensure the response structure is valid
    #         if not results.get('documents') or not results['documents'][0]:
    #             raise ValueError("No results found for the query.")

    #         # Structure results with Book # and Hadees #
    #         structured_results = []
    #         for i in range(len(results['documents'][0])):
    #             metadata = results['metadatas'][0][i]
    #             structured_results.append({
    #                 'book_number': metadata.get("book_number", "Unknown"),
    #                 'hadees_number': metadata.get("hadees_number", "Unknown"),
    #                 'text': results['documents'][0][i],
    #                 'relevance_score': 1 - results['distances'][0][i]
    #             })

    #         return structured_results

    #     except Exception as e:
    #         logger.error(f"Error querying collection: {str(e)}")
    #         raise



    
    def process_video_query(self, video_url: str, n_results: int = 3) -> list:
        """
        Process a video query by downloading, transcribing, and querying the collection.
        """
        try:
            # Step 1: Download the video
            video_path = self._download_video(video_url)

            # Step 2: Extract audio
            audio_path = self._extract_audio(video_path)

            # Step 3: Transcribe audio to text using Groq API
            transcribed_text = self._transcribe_audio_groq(audio_path)

            # Step 4: Query with the transcribed text
            results = self.query(transcribed_text, n_results=n_results)

            return results

        except Exception as e:
            print(f"Error processing video query: {str(e)}")
            raise

    def _download_video(self, video_url: str) -> str:
        """
        Download video from the given URL.
        """
        output_path = "downloaded_video.mp4"
        ydl_opts = {
            'format': 'bestvideo+bestaudio/best',
            'outtmpl': output_path,
            'quiet': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        return output_path

    def _extract_audio(self, video_path: str) -> str:
        """
        Extract audio from the downloaded video.
        """
        audio_path = "extracted_audio.m4a"
        video_clip = AudioFileClip(video_path)
        video_clip.audio.write_audiofile(audio_path, codec="aac")
        video_clip.close()
        return audio_path

    def _transcribe_audio_groq(self, audio_path: str) -> str:
        """
        Transcribe audio to text using Groq's Whisper-Large-v3 API.
        """
        client = Groq()
        with open(audio_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(audio_path, file.read()),
                model="whisper-large-v3",
                response_format="verbose_json",
            )
        return transcription['text']

    def download_video(self ,video_url, output_file="downloaded_video.mp4"):
        ydl_opts = {"outtmpl": output_file}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        return output_file

    # Function to extract audio from video
    def extract_audio(self,video_path, audio_path="extracted_audio.m4a"):
        clip = AudioFileClip(video_path)
        clip.write_audiofile(audio_path, codec="aac")
        return audio_path

    # Function to transcribe audio to text
    def transcribe_audio(self,audio_path):
        client = Groq()
        with open(audio_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(audio_path, file.read()),
                model="whisper-large-v3",
                response_format="verbose_json",
            )
        return transcription.text


    #Main pipeline function
    # def video_to_query_pipeline(self,video_url):
    #     try:
    #         print("Downloading video...")
    #         video_path = self.download_video(video_url)

    #         print("Extracting audio...")
    #         audio_path = self.extract_audio(video_path)

    #         print("Transcribing audio...")
    #         transcription_text = self.transcribe_audio(audio_path)
    #         print("Transcription Result:", transcription_text)

    #         print("Processing query...")
    #         query_result = self.query(transcription_text)
    #         print("Query Result:", query_result)

    #     except Exception as e:
    #         print("An error occurred:", e)





    def video_to_query_pipeline(self, video_url):
        try:
            print(f"Attempting to process video: {video_url}")

            # Download video
            try:
                video_path = self.download_video(video_url)
                print(f"Video downloaded successfully: {video_path}")
            except Exception as download_error:
                print(f"Video download failed: {download_error}")
                return None

            # Extract audio
            try:
                audio_path = self.extract_audio(video_path)
                print(f"Audio extracted successfully: {audio_path}")
            except Exception as audio_error:
                print(f"Audio extraction failed: {audio_error}")
                return None

            # Transcribe audio
            try:
                transcription_text = self.transcribe_audio(audio_path)
                #print(f"Transcription (length: {len(transcription_text)} characters):\n{transcription_text}")
                
                # Check for empty transcription
                if not transcription_text or len(transcription_text.strip()) < 10:
                    print("Transcription is too short or empty")
                    return None
            except Exception as transcribe_error:
                print(f"Audio transcription failed: {transcribe_error}")
                return None

            # Query with transcription
            try:
                query_result = self.query(transcription_text)
                print(f"Query returned {len(query_result)} results")
                return transcription_text
            except Exception as query_error:
                print(f"Query processing failed: {query_error}")
                return None

        except Exception as e:
            print(f"Unexpected error in video query pipeline: {e}")
            return None
        
    # def video_to_query_pipeline(self, video_url):
    #     try:
    #         print(f"Attempting to process video: {video_url}")

    #         # Download video
    #         try:
    #             video_path = self.download_video(video_url)
    #             print(f"Video downloaded successfully: {video_path}")
    #         except Exception as download_error:
    #             print(f"Video download failed: {download_error}")
    #             return None

    #         # Extract audio
    #         try:
    #             audio_path = self.extract_audio(video_path)
    #             print(f"Audio extracted successfully: {audio_path}")
    #         except Exception as audio_error:
    #             print(f"Audio extraction failed: {audio_error}")
    #             return None

    #         # Transcribe audio
    #         try:
    #             transcription_text = self.transcribe_audio(audio_path)
    #             print("--- Transcribed Text ---")
    #             print(transcription_text)
    #             print("--- End of Transcription ---")
                
    #             # Check for empty transcription
    #             if not transcription_text or len(transcription_text.strip()) < 10:
    #                 print("Transcription is too short or empty")
    #                 return None
    #         except Exception as transcribe_error:
    #             print(f"Audio transcription failed: {transcribe_error}")
    #             return None

    #         # Query with transcription
    #         try:
    #             query_result = self.query(transcription_text)
    #             print(f"Query returned {len(query_result)} results")
    #             return transcription_text
    #         except Exception as query_error:
    #             print(f"Query processing failed: {query_error}")
    #             return None

    #     except Exception as e:
    #         print(f"Unexpected error in video query pipeline: {e}")
    #         return None

    # def video_to_query_pipeline(self, video_url):
    #     try:
    #         # Download video
    #         try:
    #             video_path = self.download_video(video_url)
    #             print(f"Video downloaded successfully: {video_path}")
    #         except Exception as download_error:
    #             print(f"Video download failed: {download_error}")
    #             return None

    #         # Extract audio
    #         try:
    #             audio_path = self.extract_audio(video_path)
    #             print(f"Audio extracted successfully: {audio_path}")
    #         except Exception as audio_error:
    #             print(f"Audio extraction failed: {audio_error}")
    #             return None

    #         # Transcribe audio
    #         try:
    #             transcription_text = self.transcribe_audio(audio_path)
    #             print(f"Transcription (length: {len(transcription_text)} characters)")
                
    #             # Check for empty transcription
    #             if not transcription_text or len(transcription_text.strip()) < 10:
    #                 print("Transcription is too short or empty")
    #                 return None
    #         except Exception as transcribe_error:
    #             print(f"Audio transcription failed: {transcribe_error}")
    #             return None

    #         # Query with transcription
    #         try:
    #             query_result = self.query(transcription_text)
    #             print(f"Query returned {len(query_result)} results")
    #             return transcription_text
    #         except Exception as query_error:
    #             print(f"Query processing failed: {query_error}")
    #             return None

    #     except Exception as e:
    #         print(f"Unexpected error in video query pipeline: {e}")
    #         return None
        

    # def query(self, query_text: str, n_results: int = 6) -> List[Dict[str, str]]:
    #     try:
    #         # Flatten and ensure single-level embedding
    #         with torch.no_grad():
    #             query_embedding = self.get_model().encode(query_text)
                
    #             # Ensure the embedding is a 1D list of floats
    #             if isinstance(query_embedding, np.ndarray):
    #                 query_embedding = query_embedding.tolist()
                
    #             # Flatten any nested lists
    #             if isinstance(query_embedding[0], list):
    #                 query_embedding = [float(val) for sublist in query_embedding for val in sublist]

    #         results = self.collection.query(
    #             query_embeddings=[query_embedding],
    #             n_results=n_results,
    #             include=["documents", "metadatas", "distances"]
    #         )

    #         # Ensure the response structure is valid
    #         if not results.get('documents') or not results['documents'][0]:
    #             raise ValueError("No results found for the query.")

    #         # Structure results with Book # and Hadees #
    #         structured_results = []
    #         for i in range(len(results['documents'][0])):
    #             metadata = results['metadatas'][0][i]
    #             structured_results.append({
    #                 'book_number': metadata.get("book_number", "Unknown"),
    #                 'hadees_number': metadata.get("hadees_number", "Unknown"),
    #                 'text': results['documents'][0][i],
    #                 'relevance_score': 1 - results['distances'][0][i]
    #             })

    #         return structured_results

    #     except Exception as e:
    #         logger.error(f"Error querying collection: {str(e)}")
    #         raise