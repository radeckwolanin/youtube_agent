'''
Transcription utilities that generate textual summaries of Youtube videos, given their URL(s)
- yt_get uses pytube to download the video URLs into a local file
- yt_transcribe used the Whisper ASR model to convert the audio into text

'''

import whisper
import datetime
import subprocess
from pathlib import Path
import pandas as pd
import re
import time
import os 
import numpy as np

from pytube import YouTube
import time

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