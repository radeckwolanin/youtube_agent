from langchain.tools import BaseTool
from youtube_search import YoutubeSearch
from langchain.document_loaders import YoutubeLoader
import json

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

        with open("transcriptions.json", "w") as json_file:
            json.dump(transcriptions, json_file)
            
        return transcriptions
    
    def _run(self, query: str) -> str:
        """Use the tool."""
        return self._transcribe(query)
    
    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("YTSS  does not yet support async")