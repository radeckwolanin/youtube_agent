#Import things that are needed generically
import os
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain import LLMMathChain, SerpAPIWrapper
from langchain.agents import AgentType, Tool, initialize_agent, tool
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool
from langchain.document_loaders import YoutubeLoader

from typing import Type
from youtube_search import YoutubeSearch
import json
#from yt_utils import yt_get, yt_transcribe

load_dotenv() # Load environment variables from .env file
#OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # Get API key from environment variable - not needed since load_dotenv()

'''
CustomYTSearchTool searches YouTube videos and returns a specified number of video URLs.
Input to this tool should be a comma separated list,
 - the first part contains a subject
 - and the second(optional) a number that is the maximum number of video results to return
'''
class CustomYTSearchTool(BaseTool): 
    name = "CustomYTSearch"
    description = "search for youtube videos. the input to this tool should be a comma separated list, the first part contains a search query and the second a number that is the maximum number of video results to return aka num_results. the second part is optional"

    def _search(self, subject:str, num_results) -> str:
        results = YoutubeSearch(subject,num_results).to_json()
        data = json.loads(results)
        url_suffix_list = [video['url_suffix'] for video in data['videos']]
        return url_suffix_list
    
    def _run(self, query: str) -> str:
        """Use the tool."""
        values = query.split(",")
        subject = values[0]
        if len(values)>1:
            num_results = int(values[1])
        else:
            num_results=2
        return self._search(subject,num_results)
    
    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("YTSS  does not yet support async")

'''
CustomYTTranscribeTool transcribes YouTube videos and
saves the transcriptions in transcriptions.json in your current directory
'''

class CustomYTTranscribeTool(BaseTool):
    name = "CustomYTTranscribe"
    description = "transcribe youtube videos"

    def _transcribe(self, url_csv:str) -> str:
        values_list = url_csv.split(",")
        url_set = set(values_list)
        datatype = type(url_set)
        print(f"[YTTRANSCIBE***], received type {datatype} = {url_set}")

        transcriptions = {}

        for vurl in url_set:
            #vpath = yt_get(vurl)
            stripped_url = vurl.strip(" '")
            vpath = "https://youtube.com"+stripped_url
                      
            loader = YoutubeLoader.from_youtube_url(vpath, add_video_info=True)
            result = loader.load()
            
            transcription = result[0].page_content
            
            transcriptions[vurl]=transcription

            print(f"transcribed {vpath} into :\n {transcription}")

        with open("transcriptions.json", "w") as json_file:
            json.dump(transcriptions, json_file)
            
        return
    
    def _run(self, query: str) -> str:
        """Use the tool."""
        return self._transcribe(query)
    
    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("YTSS  does not yet support async")

'''
SummarizationTool summarizes any text and saves it to the file.
'''

class SummarizationTool(BaseTool):
    name = "SummarizationTool"
    description = "summarizes any text document. The input to this tool should be name of the json file that contains text to be summarized."

    def _summarize(self, url_csv:str) -> str:
        values_list = url_csv.split(",")
        url_set = set(values_list)
        datatype = type(url_set)
        print(f"[YTTRANSCIBE***], received type {datatype} = {url_set}")

        transcriptions = {}

        for vurl in url_set:
            vpath = yt_get(vurl)
                      
            loader = YoutubeLoader.from_youtube_url(vpath, add_video_info=True)
            result = loader.load()
            
            transcription = result[0].page_content
            
            transcriptions[vurl]=transcription

            print(f"transcribed {vpath} into :\n {transcription}")

        with open("transcriptions.json", "w") as json_file:
            json.dump(transcriptions, json_file)
            
        return
    
    def _run(self, query: str) -> str:
        """Use the tool."""
        return self._summarize(query)
    
    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("SummarizationTool  does not yet support async")

if __name__ == "__main__":
    llm = OpenAI(temperature=0)
    tools = []

    tools.append(CustomYTSearchTool())
    tools.append(CustomYTTranscribeTool())
    tools.append(SummarizationTool())
    
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    #agent.run("search youtube for Elon Musk youtube videos, and return upto 3 results. list out the results for  video URLs. for each url_suffix in the search JSON output transcribe the youtube videos")
    agent.run("use transcription from transcriptions.json and summarize it")
