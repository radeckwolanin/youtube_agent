import os
import json
from langchain.tools import BaseTool
from langchain.document_loaders import YoutubeLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from youtube_search import YoutubeSearch
import chromadb

def get_vector_store(collection_name):
    client = chromadb.HttpClient(host="20.115.73.2", port=8000)
        
    index = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=OpenAIEmbeddings()
    )
    return index

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

        transcriptions = {}

        for vurl in url_set:
            #vpath = yt_get(vurl)
            stripped_url = vurl.strip(" '")
            splitted_url = stripped_url.split(".com")[-1] # input can be with or without youtube.com
            vpath = "https://youtube.com"+splitted_url
                      
            loader = YoutubeLoader.from_youtube_url(vpath, add_video_info=True)
            result = loader.load()
            
            if len(result) == 0:
                print(result)
                raise NotImplementedError("YTTRANSCRIBE does not return any transcription")
            else:
                transcription = result[0].page_content            
                transcriptions[vurl]=transcription
                print(f"transcribed {vpath} into :\n {transcription}")

        with open("yt_transcriptions.json", "w") as json_file:
            json.dump(transcriptions, json_file)
            
        return "Inform user that transcriptions has been saved in yt_transcriptions.json file."
    
    def _run(self, query: str) -> str:
        """Use the tool."""
        return self._transcribe(query)
    
    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("YTSS  does not yet support async")
    

"""
TODO:
- Use RecursiveCharacterTextSplitter to split each transcript
- Run map_reduce chain to summarize each transcript
"""
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
        raise NotImplementedError("SummarizationTool  does not yet support async")

'''
SummarizationTool summarizes any text and saves it to the file.
'''
class SummarizationTool(BaseTool):
    name = "SummarizationTool"
    description = "summarizes any text document. The input to this tool should be name of the json file that contains text to be summarized. If the file name is not specified, use yt_transcriptions.json as default"

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