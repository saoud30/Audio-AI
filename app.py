import streamlit as st
import assemblyai as aai
import tempfile
import os
from datetime import datetime
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
from moviepy.editor import VideoFileClip
import logging
from yt_dlp import YoutubeDL
import re

# Load environment variables
load_dotenv()

# Get API key from environment variables
ASSEMBLY_AI_API_KEY = os.getenv('ASSEMBLY_AI_API_KEY')

# Configure AssemblyAI with API key from environment variables
aai.settings.api_key = ASSEMBLY_AI_API_KEY

class TranscriptionApp:
    def __init__(self):
        self.transcriber = aai.Transcriber()
        self.validate_api_key()
        self.setup_streamlit()
        
    def is_youtube_url(self, url):
        """Check if the URL is a YouTube video"""
        youtube_patterns = [
            r'^https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+',
            r'^https?://(?:www\.)?youtube\.com/shorts/[\w-]+',
            r'^https?://youtu\.be/[\w-]+',
        ]
        return any(re.match(pattern, url) for pattern in youtube_patterns)
    
    def is_instagram_url(self, url):
        """Check if the URL is an Instagram Reel"""
        instagram_patterns = [
            r'^https?://(?:www\.)?instagram\.com/reel/[\w-]+',
            r'^https?://(?:www\.)?instagram\.com/p/[\w-]+',
        ]
        return any(re.match(pattern, url) for pattern in instagram_patterns)
    
    def download_social_media_video(self, url):
        """Download video from YouTube or Instagram"""
        try:
            with st.spinner("Downloading video from social media..."):
                temp_video_path = tempfile.mktemp(suffix='.mp4')
                ydl_opts = {
                    'format': 'best',  # Best quality
                    'outtmpl': temp_video_path,
                    'quiet': True,
                    'no_warnings': True,
                    'extract_audio': False,  # We'll extract audio using moviepy later
                }
                
                # Add Instagram-specific cookies if it's an Instagram URL
                if self.is_instagram_url(url):
                    ydl_opts.update({
                        'cookiesfrombrowser': ('chrome',),  # Uses Chrome cookies
                    })
                
                with YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
                
                return temp_video_path
        except Exception as e:
            st.error(f"Error downloading video: {str(e)}")
            return None
    
    def validate_api_key(self):
        """Validate that API key is properly configured"""
        if not ASSEMBLY_AI_API_KEY:
            st.error("⚠️ AssemblyAI API key not found. Please check your .env file.")
            st.stop()
        elif ASSEMBLY_AI_API_KEY == "your_api_key_here":
            st.error("⚠️ Please replace the default API key in .env with your actual AssemblyAI API key.")
            st.stop()
    
    def setup_streamlit(self):
        st.set_page_config(
            page_title="Speech-to-Text Analyzer",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        st.title("🎙️ Speech-to-Text Analyzer")
        
        # Sidebar configuration
        st.sidebar.header("Settings")
        
        self.analysis_options = st.sidebar.multiselect(
            "Select Analysis Features",
            ["Speaker Labels", "Entity Detection", "Key Phrases"],
            default=["Speaker Labels"]
        )

    def extract_audio_from_video(self, video_path):
        """Extract audio from video file"""
        try:
            with st.spinner("Extracting audio from video..."):
                video = VideoFileClip(video_path)
                # Create temporary file for audio
                temp_audio_path = tempfile.mktemp(suffix='.mp3')
                video.audio.write_audiofile(temp_audio_path, logger=None)
                video.close()
                return temp_audio_path
        except Exception as e:
            st.error(f"Error extracting audio from video: {str(e)}")
            return None

    def save_uploaded_file(self, uploaded_file):
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                return tmp_file.name
        return None

    def analyze_transcript(self, transcript):
        if not transcript:
            return
        
        if transcript.error:
            st.error(f"Transcription Error: {transcript.error}")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📝 Transcript")
            st.write(transcript.text)
            
            if "Speaker Labels" in self.analysis_options and hasattr(transcript, 'utterances'):
                st.subheader("👥 Speaker Analysis")
                for utterance in transcript.utterances:
                    st.markdown(f"**Speaker {utterance.speaker}**: {utterance.text}")
        
        with col2:
            if "Entity Detection" in self.analysis_options and hasattr(transcript, 'entities'):
                st.subheader("🔍 Detected Entities")
                entities = transcript.entities
                if entities:
                    df_entities = pd.DataFrame(entities)
                    fig = px.bar(df_entities, x='entity_type', title='Entity Distribution')
                    st.plotly_chart(fig)
            
            if "Key Phrases" in self.analysis_options and hasattr(transcript, 'auto_highlights'):
                st.subheader("🗝️ Key Phrases")
                auto_highlights = transcript.auto_highlights
                if auto_highlights:
                    for highlight in auto_highlights:
                        st.markdown(f"- {highlight}")

    def run(self):
        st.info("📱 You can now process YouTube videos/shorts by pasting their URLs!")
        
        uploaded_file = st.file_uploader("Choose an audio/video file", type=['mp3', 'wav', 'ogg', 'm4a', 'mp4'])
        media_url = st.text_input("Or enter the URL of an audio/video file (Supports YouTube and direct media links)")
        
        if st.button("🚀 Process Media"):
            with st.spinner("Processing... This may take a few minutes."):
                try:
                    # Create config with selected features
                    config = aai.TranscriptionConfig(
                        speaker_labels=("Speaker Labels" in self.analysis_options)
                    )
                    
                    if uploaded_file:
                        file_path = self.save_uploaded_file(uploaded_file)
                        
                        # Handle MP4 files
                        if file_path.lower().endswith('.mp4'):
                            audio_path = self.extract_audio_from_video(file_path)
                            if audio_path:
                                transcript = self.transcriber.transcribe(
                                    audio_path,
                                    config=config
                                )
                                os.unlink(audio_path)
                            else:
                                st.error("Failed to process video file")
                                return
                        else:
                            transcript = self.transcriber.transcribe(
                                file_path,
                                config=config
                            )
                        os.unlink(file_path)
                    elif media_url:
                        # Handle social media URLs
                        if self.is_youtube_url(media_url) or self.is_instagram_url(media_url):
                            video_path = self.download_social_media_video(media_url)
                            if video_path:
                                audio_path = self.extract_audio_from_video(video_path)
                                if audio_path:
                                    transcript = self.transcriber.transcribe(
                                        audio_path,
                                        config=config
                                    )
                                    os.unlink(audio_path)
                                os.unlink(video_path)
                            else:
                                st.error("Failed to download video")
                                return
                        else:
                            # Try direct transcription for other URLs
                            transcript = self.transcriber.transcribe(
                                media_url,
                                config=config
                            )
                    else:
                        st.error("Please provide a file or URL")
                        return
                    
                    self.analyze_transcript(transcript)
                    
                    st.download_button(
                        label="Download Transcript",
                        data=transcript.text,
                        file_name=f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.error("Please make sure your file is in a supported format and is accessible.")

if __name__ == "__main__":
    app = TranscriptionApp()
    app.run()