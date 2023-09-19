#Import things that are needed generically
import os
from dotenv import load_dotenv
import json
from langchain.vectorstores import Chroma
import chromadb
import streamlit as st
from langchain.llms import OpenAI
from langchain import LLMMathChain, SerpAPIWrapper
from langchain.callbacks.streaming_stdout_final_only import (
    FinalStreamingStdOutCallbackHandler,
)
from langchain.callbacks.base import BaseCallbackHandler
from langchain.agents import AgentType, Tool, initialize_agent, tool, load_tools
from langchain.chat_models import ChatOpenAI
from langchain.schema import ChatMessage

from langchain.tools import BaseTool
from langchain.document_loaders import YoutubeLoader

from typing import Type
from youtube_search import YoutubeSearch
from yt_utils import get_vector_store #yt_get, yt_transcribe

load_dotenv() # Load environment variables from .env file
#OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # Get API key from environment variable - not needed since load_dotenv()

class MyCallbackHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token, **kwargs) -> None:
        # print every token on a new line
        print(f"#{token}#")

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

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
"""
TODO:
- Check if summary already exists in database
- Use RecursiveCharacterTextSplitter to split each transcript
- Run map_reduce chain to summarize each transcript

"""
class SummarizationTool(BaseTool):
    name = "SummarizationTool"
    description = "summarizes any text document. The input to this tool should be name of the json file that contains text to be summarized."

    def _summarize(self, input_file:str) -> str:
        
        if os.path.exists(input_file):
            try:
                with open(input_file, 'r') as file:
                    data = json.load(file)
                print("File loaded successfully as JSON:")
                
                if isinstance(data, dict):
                    # If the data is a dictionary
                    for key, value in data.items():
                        print(f"Key: {key}, Value: {value}")
                        # TODO - finish from here
                        
                        
            except json.JSONDecodeError as e:
                print(f"Error loading JSON: {e}")
                raise NotImplementedError(f"Error loading JSON: {e}")
        else:
            print(f"The file '{input_file}' does not exist.")
            raise NotImplementedError(f"SummarizationTool: File '{input_file}' does not exist.")
        
        return
    
    def _run(self, query: str) -> str:
        """Use the tool."""
        return self._summarize(query)
    
    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("SummarizationTool  does not yet support async")

if __name__ == "__main__":
    with st.sidebar:
        openai_api_key = st.text_input("OpenAI API Key", type="password")

    if "messages" not in st.session_state:
        st.session_state["messages"] = [ChatMessage(role="assistant", content="How can I help you?")]

    for msg in st.session_state.messages:
        st.chat_message(msg.role).write(msg.content)

    if prompt := st.chat_input():
        st.session_state.messages.append(ChatMessage(role="user", content=prompt))
        st.chat_message("user").write(prompt)

        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()

        with st.chat_message("assistant"):
            stream_handler = StreamHandler(st.empty())
            llm = ChatOpenAI(openai_api_key=openai_api_key, streaming=True, callbacks=[stream_handler])
            response = llm(st.session_state.messages)
            st.session_state.messages.append(ChatMessage(role="assistant", content=response.content))
    
    #llm = OpenAI(temperature=0)
    # Streaming
    #llm = OpenAI(
        #streaming=True, callbacks=[FinalStreamingStdOutCallbackHandler()], temperature=0
        #streaming=True, callbacks=[MyCallbackHandler()], temperature=0
    #)
    
    #tools = load_tools(["wikipedia", "serpapi", "llm-math"], llm=llm)
    
    #tools = []
    
    #tools.append(CustomYTSearchTool())
    #tools.append(CustomYTTranscribeTool())
    #tools.append(SummarizationTool())
    
    """
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        return_intermediate_steps=True,
    )
    """
    
    """
    agent.run(
        "It's 2023 now. How many years ago did Konrad Adenauer become Chancellor of Germany."
    )
    """
    
    """
    response = agent(
        {
            "input": "Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?"
        }
    )    
    print(response)
    """
    
    #agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    #agent.run("search youtube for Elon Musk youtube videos, and return upto 3 results. list out the results for  video URLs. for each url_suffix in the search JSON output transcribe the youtube videos")
    #agent.run("use transcription from transcriptions.json and summarize it")
    
    # WORKS
    #db = get_vector_store("you_tube")
    #query = "Are there any news related to Joe Biden?"
    #docs = db.similarity_search(query,2)
    #print(docs[0].page_content)
    #print(docs)
    
    
