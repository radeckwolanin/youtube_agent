'''
Transcription utilities that generate textual summaries of Youtube videos, given their URL(s)
- yt_get uses pytube to download the video URLs into a local file
- yt_transcribe used the Whisper ASR model to convert the audio into text

'''

#import whisper
#import datetime
#import subprocess
#from pathlib import Path
#import pandas as pd
#import re
#import time
#import os 
#import numpy as np

from langchain.vectorstores import Chroma
import chromadb
#from chromadb.utils import embedding_functions
#import uuid
from langchain.embeddings import OpenAIEmbeddings

from pytube import YouTube
import time

def get_vector_store(collection_name):
    client = chromadb.HttpClient(host="20.115.73.2", port=8000)
        
    index = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=OpenAIEmbeddings()
    )
    return index

"""
def load_model():
    return whisper.load_model("base")

model = load_model()
"""

def yt_get(yt_url):
    stripped_url = yt_url.strip(" '")
    return "https://youtube.com"+stripped_url
"""
def yt_transcribe(video_url):
    print(f"transcribing {video_url}")
    result = model.transcribe(video_url)
    return (result['text'])
"""