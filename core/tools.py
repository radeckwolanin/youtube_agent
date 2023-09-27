import os
import json
import datetime
#import pickle
from langchain.tools import BaseTool
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import create_extraction_chain
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

#from langchain.vectorstores import ZepVectorStore
#from langchain.vectorstores.zep import CollectionConfig
#from zep_python import ZepClient
#from langchain.embeddings import FakeEmbeddings
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings


from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from youtube_search import YoutubeSearch
import chromadb
from chromadb.utils import embedding_functions
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

def get_vector_store(collection_name: str):    
    """
    # Zep VectorStore implementation. Mising `get` command which brakes code
    # Collection config is needed if we're creating a new Zep Collection
    config = CollectionConfig(
        name=collection_name,
        description="YouTube Agent vectorstore",
        metadata={"created_by": "yt_agent"},
        is_auto_embedded=True,  # we'll have Zep embed our documents using its low-latency embedder
        embedding_dimensions=384  # this should match the model you've configured Zep to use.
    )    
    index = ZepVectorStore(
        collection_name=collection_name,
        config=config,
        api_url=os.environ.get("ZEP_API_URL"),
        api_key=os.environ.get("ZEP_API_KEY")
    )
    """
    client = chromadb.HttpClient(
        host=os.environ.get("DB_HOST"), 
        port=os.environ.get("DB_PORT")
    )
    
    # create the open-source embedding function
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    # OpenAI embedding function
    #embedding_function = OpenAIEmbeddings()
        
    index = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embedding_function
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
        #url_suffix_list = [video['url_suffix'] for video in data['videos']]
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
VectorDBCollectionAdd: Store YouTube transcripts in Chroma to later use for retrival
TODO: Use Default Sentance Transformer instead of OpenAI ?

'''
class VectorDBCollectionAdd(BaseTool):
    name = "VectorDBCollectionAdd"
    description = "stores, saves, adds text file to vector database. input to this tool is a name of the file that contains text. if no impot can be retrived from user input, use yt_transcriptions.json which contains latest transcript."

    def _collectionAdd(self, input_file:str) -> str:
        print(f"[VectorDBCollectionAdd***], File {input_file}")  
              
        if os.path.exists(input_file):
            try:
                with open(input_file, 'r', encoding='utf-8') as file:
                    loaded_serializable_file = json.load(file)
                    
                to_return = ""
                
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)  
                
                vectorstore = get_vector_store("videos")
                vectorstore_topics = get_vector_store("topics")
                vectorstore_summaries = get_vector_store("summaries")
                
                loaded_files = []   # page_content of files to be uploaded to vectorstore
                topics={}           # extracted topics to be uploaded if exists
                summaries = {}      # summaries to be uploaded if exists

                for doc_dict in loaded_serializable_file:
                    # Reconstruct Document objects from dictionaries
                    source = doc_dict['metadata']['source']
                    
                    # Save TOPICS if exists to you_tube_topics collection
                    if doc_dict['metadata']['topics']:
                        # Check if not exists in collection
                        number_of_ids = len(vectorstore_topics.get(where = {"source":source})["ids"])
                        if number_of_ids > 0:                            
                            to_return += f"- Topics for source {source} was not saved because they are alaready in database with {number_of_ids} IDs\n"
                            # Iterate over to double check if we have them all?
                        else:
                            topics[source] = doc_dict['metadata']['topics']
                            print(f"Topics for source {source} loaded: ",len(doc_dict['metadata']['topics']))                        
                        # Reset topics to just a number of topics since mtedata cannot contain list, thus save into separate collection
                        doc_dict['metadata']['topics'] = len(doc_dict['metadata']['topics'])
                    
                    # Save SUMMARY if exists to you_tube_summaries collection
                    if doc_dict['metadata']['summary']:
                        # Check if not exists in collection
                        number_of_ids = len(vectorstore_summaries.get(where = {"source":source})["ids"])
                        if number_of_ids > 0:                            
                            to_return += f"- Summary for source {source} was not saved because it is alaready in database with {number_of_ids} IDs\n"
                            # Iterate over to double check if we have them all?
                        else:
                            summaries[source] = doc_dict['metadata']['summary']
                            print(f"Summary for source {source} loaded")
                        # Reset summary to True/False indicating if summary is created 
                        doc_dict['metadata']['summary'] = True
                        
                    # Add when file was added to collection    
                    doc_dict['metadata']['added_date_time'] = datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
                    # Create document
                    doc = Document(page_content=doc_dict['page_content'], metadata=doc_dict['metadata'])
                    
                    # Check if transcript is not already in database
                    number_of_ids = len(vectorstore.get(where = {"source":source})["ids"])
                    if number_of_ids > 0:                        
                        to_return += f"- Transcript for source {source} was not saved because it is alaready in database with {number_of_ids} IDs\n"
                    else:
                        loaded_files.append(doc)
                        print("Loaded file for upload: ",doc.metadata)
                    
                
                # Check if any files to upload    
                if len(loaded_files) > 0:
                    # Split into chunks if too long text
                    splitted_texts =  text_splitter.split_documents(loaded_files)
                    id_list = vectorstore.add_documents(splitted_texts)
                    number_of_ids = len(id_list)
                    to_return += f"- Transcript for source {source} saved to database with {number_of_ids} IDs.\n"
                    
                # Check if any topics to upload    
                if len(topics) > 0:
                    for source in topics:
                        # Get metadata from source
                        collection = vectorstore.get(where = {"source":source})
                        metadata = collection['metadatas'][0] # Select first metadata since all should be the same
                        documents = []
                        idx=1
                        for topic in topics[source]:
                            # TODO: save topic_name and tag 
                            metadata['topic_num'] = idx; idx += 1
                            metadata['topic_name'] = topic['topic_name']
                            metadata['tag'] = topic['tag']
                            doc = Document(page_content=topic['description'], metadata=metadata.copy())
                            documents.append(doc)
                            
                        id_list = vectorstore_topics.add_documents(documents)
                        #print(f"ID List: {id_list}")
                    
                    number_of_ids = len(topics) #temp
                    to_return += f"- Topic for source {source} saved to database with {number_of_ids} IDs.\n"
                    
                # Check if summary to upload    
                if len(summaries) > 0:
                    for source in summaries:
                        # Get metadata from source
                        collection = vectorstore.get(where = {"source":source})
                        metadata = collection['metadatas'][0] # Select first metadata since all should be the same
                        documents = []
                        doc = Document(page_content=summaries[source], metadata=metadata.copy())
                        documents.append(doc)
                        id_list = vectorstore_summaries.add_documents(documents)
                        #print(f"ID List: {id_list}")
                    
                    number_of_ids = len(summaries) #temp
                    to_return += f"- Summary for source {source} saved to database with {number_of_ids} IDs.\n"
                
                #number_of_ids = len(vectorstore.get(where = {"title":temp_title})["ids"])
                #print(f"Number of ids stored {number_of_ids}")
                print(to_return)
                return f"Here is a summary:\n {to_return}"
                        
            except json.JSONDecodeError as e:
                print(f"Error loading JSON: {e}")
                raise NotImplementedError(f"Error loading JSON: {e}")
        else:
            print(f"The file '{input_file}' does not exist.")
            raise NotImplementedError(f"VectorDBCollectionAdd: File '{input_file}' does not exist.")
    
    def _run(self, query: str) -> str:
        """Use the tool."""
        return self._collectionAdd(query)
    
    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("VectorDBCollectionAdd does not yet support async")
    
'''
SummarizationTool summarizes any text and saves it to the file.
TODO: Return all summaries, not only from the first link
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
                
                # Creating two versions of the model so I can swap between gpt3.5 and gpt4
                llm3 = ChatOpenAI(temperature=0,
                                model_name="gpt-3.5-turbo-0613",
                                request_timeout = 180
                                )

                llm4 = ChatOpenAI(temperature=0,
                                model_name="gpt-4-0613",
                                request_timeout = 180
                                )
                chain = load_summarize_chain(llm4, chain_type="map_reduce", verbose=False)
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