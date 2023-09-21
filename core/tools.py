import os
import json
#import pickle
from langchain.tools import BaseTool
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import create_extraction_chain
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from youtube_search import YoutubeSearch
import chromadb
from pydantic import BaseModel, Field

from core.prompts import (
    CHAT_PROMPT_MAP, 
    CHAT_PROMPT_COMBINE,
    CHAT_PROMPT_EXPAND,
)

"""
TODO:
- Save transcript to database
- Use RetrivalQA from vectorstore to expand on each topic
- Based on extracted data, create new content/tweet
"""

def get_vector_store(collection_name):
    client = chromadb.HttpClient(host="20.115.73.2", port=8000)
        
    index = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=OpenAIEmbeddings()
    )
    return index

class Document(BaseModel):
    """Interface for interacting with a document."""

    page_content: str
    metadata: dict = Field(default_factory=dict)

    def to_dict(self):
        return self.dict(by_alias=True, exclude_unset=True) # just an example!

    def to_json(self):
        return self.json(by_alias=True, exclude_unset=True) # just an example!

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
        #return url_suffix_list
        with open("yt_search.json", "w") as json_file:
            json.dump(results, json_file)
            
        return results
    
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
    description = "transcribe youtube videos. input to this tool is a comma separated list of URLs"

    def _transcribe(self, url_csv:str) -> str:
        values_list = url_csv.split(",")
        url_set = set(values_list)
        datatype = type(url_set)
        print(f"[YTTRANSCIBE***], received type {datatype} = {url_set}")
        
        serializable_transcriptions = []

        for vurl in url_set:
            stripped_url = vurl.strip(" '")
            #splitted_url = stripped_url.split(".com")[-1] # input can be with or without youtube.com
            source = stripped_url.split("watch?v=")[-1] # input can be with or without youtube.com
            vpath = "https://youtube.com/watch?v="+source
            #vpath = "https://youtube.com"+splitted_url
                                  
            loader = YoutubeLoader.from_youtube_url(vpath, add_video_info=True)
            result = loader.load()
            
            if len(result) == 0:
                print(result)
                raise NotImplementedError("YTTRANSCRIBE does not return any transcription")
            else:
                doc_dict = {
                    'page_content': result[0].page_content,
                    'metadata': result[0].metadata
                }
                serializable_transcriptions.append(doc_dict)
                print(f"Transcribed {vpath} Transcription: {result[0].page_content[:50]}...")
        
        with open('yt_transcriptions.json', 'w', encoding='utf-8') as file:
            json.dump(serializable_transcriptions, file, ensure_ascii=False, indent=4)
            
        return "Inform user that transcriptions has been saved in yt_transcriptions.json file."
    
    def _run(self, query: str) -> str:
        """Use the tool."""
        return self._transcribe(query)
    
    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("YTSS  does not yet support async")
    

'''
VectorDBCheckStatus checks if given youtube url is already in VectorDB you_tube collection

'''
class VectorDBCheckStatus(BaseTool):
    name = "VectorDBCheckStatus"
    description = "checks status if given video transcript already exists in vector database. input to this tool is a comma separated list of URLs to you tube video. if no impot is present, use yt_transcriptions.json"

    def _checkStatus(self, url_csv:str) -> str:
        values_list = url_csv.split(",")
        url_set = set(values_list)
        datatype = type(url_set)
        print(f"[VectorDBCheckStatus***], received type {datatype} = {url_set}")
        
        db_status = {}
        vectorstore = get_vector_store("you_tube")
        
        for vurl in url_set:
            stripped_url = vurl.strip(" '")
            source = stripped_url.split("watch?v=")[-1] # input can be with or without youtube.com     
            number_of_ids = len(vectorstore.get(where = {"source":source})["ids"])
            
            db_status[vurl]={"SOURCE": source, "NUMBER OF RECORDS": number_of_ids}
            #print(db_status)            
            #check if this link https://www.youtube.com/watch?v=piYf4gDthjY is already in database
            #check if this link https://www.youtube.com/watch?v=UfL7hqGBLAQ is already in database
        return db_status
    
    def _run(self, query: str) -> str:
        """Use the tool."""
        return self._checkStatus(query)
    
    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("VectorDBCheckStatus does not yet support async")

'''
SummarizationTool summarizes any text and saves it to the file.
'''
class SummarizationTool(BaseTool):
    name = "SummarizationTool"
    description = "summarizes any text document. The input to this tool should be name of the json file that contains text to be summarized. If the file name is not specified, use yt_transcriptions.json as default"

    def _summarize(self, input_file:str) -> str:
        
        if os.path.exists(input_file):
            try:
                with open(input_file, 'r', encoding='utf-8') as file:
                    loaded_serializable_transcriptions = json.load(file)
                
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)         
                
                # Reconstruct Document objects from the loaded data
                loaded_transcriptions = []
                summaries = []

                for doc_dict in loaded_serializable_transcriptions:
                    # Reconstruct Document objects from dictionaries
                    doc = Document(page_content=doc_dict['page_content'], metadata=doc_dict['metadata'])
                    print("Loaded transcript: ",doc.metadata)
                    loaded_transcriptions.append(doc)
                
                    # Split into chunks if too long
                    splitted_transcriptions =  text_splitter.split_documents(loaded_transcriptions)
                    
                    chain = load_summarize_chain(OpenAI(temperature=0), chain_type="map_reduce", verbose=False)
                    doc.metadata['summary'] = chain.run(splitted_transcriptions)                    
                    summaries.append(doc.to_dict())
                
                with open('yt_transcriptions.json', 'w', encoding='utf-8') as file:
                    json.dump(summaries, file, ensure_ascii=False, indent=4)
                    
                # TODO: 
                # - Return summaries of each video, not only first [0]
                print(summaries[0]['metadata']['summary'])
                return f"SUMMARY: {summaries[0]['metadata']['summary']}"
                        
            except json.JSONDecodeError as e:
                print(f"Error loading JSON: {e}")
                raise NotImplementedError(f"Error loading JSON: {e}")
        else:
            print(f"The file '{input_file}' does not exist.")
            raise NotImplementedError(f"SummarizationTool: File '{input_file}' does not exist.")
        
        return "Summary of the transcript: this video talks about iPhone 15"
    
    def _run(self, query: str) -> str:
        """Use the tool."""
        return self._summarize(query)
    
    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("SummarizationTool  does not yet support async")
    
'''
ExtractInfoTool extracts information from transcript
'''
class ExtractInfoTool(BaseTool):
    name = "ExtractInfoTool"
    description = "extracts valuable information from any text document like topics and short description of each topic. The input to this tool should be name of the json file that contains text. If the file name is not specified, use yt_transcriptions.json as default where latest transcript is saved."

    def _extractInfo(self, input_file:str) -> str:
        
        if os.path.exists(input_file):
            try:
                with open(input_file, 'r', encoding='utf-8') as file:
                    loaded_serializable_transcriptions = json.load(file)
                
                # Creating two versions of the model so I can swap between gpt3.5 and gpt4
                llm3 = ChatOpenAI(temperature=0,
                                model_name="gpt-3.5-turbo-0613",
                                request_timeout = 180
                                )

                llm4 = ChatOpenAI(temperature=0,
                                model_name="gpt-4-0613",
                                request_timeout = 180
                                )
                # use llm4 for extraction, llm3 just for testing
                chain = load_summarize_chain(llm3, # llm4
                             chain_type="map_reduce",
                             map_prompt=CHAT_PROMPT_MAP,
                             combine_prompt=CHAT_PROMPT_COMBINE,
                             verbose=False
                            )                
                
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)         
                
                # Reconstruct Document objects from the loaded data
                loaded_transcriptions = []
                topics_return = []

                for doc_dict in loaded_serializable_transcriptions:
                    # Reconstruct Document objects from dictionaries
                    doc = Document(page_content=doc_dict['page_content'], metadata=doc_dict['metadata'])
                    print("Loaded transcript: ",doc.metadata)
                    loaded_transcriptions.append(doc)
                
                    # Split into chunks if too long
                    splitted_transcriptions =  text_splitter.split_documents(loaded_transcriptions)
                    topics_found = chain.run({"input_documents": splitted_transcriptions})
                                        
                    # Create schema for topic extraction
                    schema = {
                        "properties": {
                            # The title of the topic
                            "topic_name": {
                                "type": "string",
                                "description" : "The title of the topic listed"
                            },
                            # The description
                            "description": {
                                "type": "string",
                                "description" : "The description of the topic listed"
                            },
                            "tag": {
                                "type": "string",
                                "description" : "The type of content being described",
                                "enum" : ['Business Models', 'Life Advice', 'Health & Wellness', 'Stories', 'Politics']
                            }
                        },
                        "required": ["topic", "description"],
                    }
                    
                    chain = create_extraction_chain(schema, llm3)
                    topics_structured = chain.run(topics_found)
                    
                    doc.metadata['topics'] = topics_structured
                    topics_return.append(doc.to_dict())
                    
                with open('yt_transcriptions.json', 'w', encoding='utf-8') as file:
                    json.dump(topics_return, file, ensure_ascii=False, indent=4)
                    
                # TODO: 
                # - Return extraction of each video, not only first [0]
                print(topics_return[0]['metadata']['topics'])
                return f"TOPICS DISCUSSED: {topics_return[0]['metadata']['topics']}"
                        
            except json.JSONDecodeError as e:
                print(f"Error loading JSON: {e}")
                raise NotImplementedError(f"Error loading JSON: {e}")
        else:
            print(f"The file '{input_file}' does not exist.")
            raise NotImplementedError(f"ExtractInfoTool: File '{input_file}' does not exist.")
    
    def _run(self, query: str) -> str:
        """Use the tool."""
        return self._extractInfo(query)
    
    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("ExtractInfoTool  does not yet support async")