import streamlit as st
import torch
from sentence_transformers import SentenceTransformer
import chromadb
import os
import logging
from typing import List, Dict
from client import IslamicSourceFinderClient
import atexit
import tempfile
from contextlib import contextmanager

class IslamicSourceFinderApp:
    def __init__(self):
        # Initialize the client with the persist directory
        self.finder = IslamicSourceFinderClient(persist_directory="~/chromadb_data")
        self.temp_files = set()  # Track temporary files
        atexit.register(self.cleanup_temp_files)  # Register cleanup on exit
        
    def cleanup_temp_files(self):
        """Clean up all temporary files"""
        for temp_path in self.temp_files:
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    print(f"Cleaned up temporary file: {temp_path}")
            except Exception as e:
                print(f"Error cleaning up {temp_path}: {e}")
        self.temp_files.clear()

    @contextmanager
    def temporary_file(self, uploaded_file):
        """Context manager for handling temporary files"""
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"temp_{uploaded_file.name}")
        try:
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            self.temp_files.add(temp_path)  # Track this temp file
            yield temp_path
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                    self.temp_files.remove(temp_path)
                except Exception as e:
                    print(f"Error removing temporary file {temp_path}: {e}")

    def run(self):
        st.title("Islamic Source Finder")
        
        # Sidebar for navigation
        app_mode = st.sidebar.selectbox(
            "Choose Application Mode",
            ["Text Query", "Video Query", "Image Query", "Document Processing"]
        )
        
        if app_mode == "Text Query":
            self.text_query_interface()
        elif app_mode == "Video Query":
            self.video_query_interface()
        elif app_mode == "Image Query":
            self.image_query_interface()
        elif app_mode == "Document Processing":
            self.document_processing_interface()
    
    def text_query_interface(self):
        st.header("Text-based Query")
        query = st.text_area("Enter your query:", height=100)
        n_results = st.slider("Number of results", min_value=1, max_value=10, value=3)
        
        if st.button("Search"):
            if query:
                try:
                    with st.spinner("Searching..."):
                        results = self.finder.query(query, n_results=n_results)
                    
                    if results:
                        st.write(f"Found {len(results)} relevant passages:")
                        for result in results:
                            with st.expander(f"Book {result['book_number']}, Hadees {result['hadees_number']} (Relevance: {result['relevance_score']:.2f})"):
                                st.write(result['text'])
                    else:
                        st.warning("No results found.")
                except Exception as e:
                    st.error(f"Error processing query: {e}")
            else:
                st.warning("Please enter a query.")
    
    def video_query_interface(self):
        st.header("Video Query")
        video_url = st.text_input("Enter Video URL:")
        
        if st.button("Process Video"):
            if video_url:
                try:
                    with st.spinner("Processing video..."):
                        transcription = self.finder.video_to_query_pipeline(video_url)
                    
                    if transcription:
                        st.success("Video processed successfully!")
                        st.subheader("Transcription:")
                        st.write(transcription)
                        
                        results = self.finder.query(transcription)
                        if results:
                            st.subheader("Relevant Passages:")
                            for result in results:
                                with st.expander(f"Book {result['book_number']}, Hadees {result['hadees_number']} (Relevance: {result['relevance_score']:.2f})"):
                                    st.write(result['text'])
                    else:
                        st.warning("Could not process video.")
                except Exception as e:
                    st.error(f"Error processing video: {e}")
            else:
                st.warning("Please enter a video URL.")
    
    def image_query_interface(self):
        st.header("Image Query")
        uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])
        
        if uploaded_file is not None:
            if st.button("Process Image"):
                try:
                    # Use context manager for temporary file handling
                    with self.temporary_file(uploaded_file) as temp_path:
                        with st.spinner("Processing image..."):
                            extracted_text = self.finder.groq_ocr(temp_path)
                        
                        if extracted_text:
                            st.success("Image processed successfully!")
                            st.subheader("Extracted Text:")
                            st.write(extracted_text)
                            
                            results = self.finder.query(extracted_text)
                            if results:
                                st.subheader("Relevant Passages:")
                                for result in results:
                                    with st.expander(f"Book {result['book_number']}, Hadees {result['hadees_number']} (Relevance: {result['relevance_score']:.2f})"):
                                        st.write(result['text'])
                except Exception as e:
                    st.error(f"Error processing image: {e}")
    
    def document_processing_interface(self):
        st.header("Document Processing")
        
        # Add a button to clean up temp files
        col1, col2 = st.columns([3, 1])
        with col1:
            uploaded_files = st.file_uploader(
                "Upload PDF or Text Files", 
                type=['pdf', 'txt'], 
                accept_multiple_files=True
            )
        with col2:
            if st.button("Clean Temp Files"):
                cleaned_count = len(self.temp_files)
                self.cleanup_temp_files()
                if cleaned_count > 0:
                    st.success(f"Cleaned up {cleaned_count} temporary file(s)")
                else:
                    st.info("No temporary files to clean")
        
        if st.button("Process Documents"):
            if uploaded_files:
                processed_count = 0
                for uploaded_file in uploaded_files:
                    try:
                        # Use context manager for temporary file handling
                        with self.temporary_file(uploaded_file) as temp_path:
                            with st.spinner(f"Processing {uploaded_file.name}..."):
                                self.finder.process_text_file(temp_path)
                                processed_count += 1
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {e}")
                
                if processed_count > 0:
                    st.success(f"Successfully processed {processed_count} document(s)")
            else:
                st.warning("Please upload documents.")

def main():
    app = IslamicSourceFinderApp()
    app.run()

if __name__ == "__main__":
    main() 