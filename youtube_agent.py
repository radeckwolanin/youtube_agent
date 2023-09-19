import os
from dotenv import load_dotenv
from langchain.agents import ConversationalChatAgent, AgentType, AgentExecutor, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.tools import DuckDuckGoSearchRun
import streamlit as st
from core.tools import CustomYTSearchTool

load_dotenv() # Load environment variables from .env file

st.set_page_config(page_title="YouTube Agent", page_icon="📺")
st.title("📺 YouTube Agent")
st.markdown('Welcome to YouTube Personal Assistant. Provide any subject that you want to explore using YouTube.')

msgs = StreamlitChatMessageHistory()

memory = ConversationBufferMemory(
    chat_memory=msgs, 
    return_messages=True, 
    memory_key="chat_history", 
    output_key="output"
)

if len(msgs.messages) == 0: #or st.sidebar.button("Reset chat history"):
    msgs.clear()
    msgs.add_ai_message("What YouTube videos do you want me to search for?")
    st.session_state.steps = {}

avatars = {"human": "user", "ai": "assistant"}
for idx, msg in enumerate(msgs.messages):
    with st.chat_message(avatars[msg.type]):
        # Render intermediate steps if any were saved
        for step in st.session_state.steps.get(str(idx), []):
            if step[0].tool == "_Exception":
                continue
            with st.status(f"**{step[0].tool}**: {step[0].tool_input}", state="complete"):
                st.write(step[0].log)
                st.write(step[1])
        st.write(msg.content)

if prompt := st.chat_input(placeholder="Todays top global news"):
    st.chat_message("user").write(prompt)

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", streaming=True)
    
    #tools = [DuckDuckGoSearchRun(name="Search")]
    tools = []
    tools.append(CustomYTSearchTool())
    
    #chat_agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=False)
    chat_agent = ConversationalChatAgent.from_llm_and_tools(llm=llm, tools=tools)
    
    # The system instructions. Notice the 'context' placeholder down below. This is where our relevant docs will go.
    # The 'question' in the human message below won't be a question per se, but rather a topic we want to get relevant information on
    system_template = """
    You will be given a YouTube search query. 
    Your goal is to search YouTube for related videos and return 3 links.
    Lastly, end final answer with some funny quote.
    ----------------
    {prompt}"""

    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(prompt),
    ]

    # This will pull the two messages together and get them ready to be sent to the LLM through the retriever
    CHAT_PROMPT = ChatPromptTemplate.from_messages(messages)
    
    executor = AgentExecutor.from_agent_and_tools(
        agent=chat_agent,
        tools=tools,
        memory=memory,
        return_intermediate_steps=True,
        handle_parsing_errors=True
    )
    
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        
        #agent.run("search youtube for Elon Musk youtube videos, and return upto 3 results. list out the results for video URLs.")
        #agent.run("search youtube for Elon Musk youtube videos, and return upto 3 results. list out the results for  video URLs. for each url_suffix in the search JSON output transcribe the youtube videos")
        #agent.run("use transcription from transcriptions.json and summarize it")
        
        new_prompt = f"search youtube for {prompt} videos, and return upto 3 results. list out the results for video URLs."
        print(new_prompt)
        #response = executor(prompt, callbacks=[st_cb])
        response = executor(new_prompt, callbacks=[st_cb])
        
        #agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
        #agent.run("search youtube for Elon Musk youtube videos, and return upto 3 results. list out the results for  video URLs. for each url_suffix in the search JSON output transcribe the youtube videos")
        
        st.write(response["output"])
        print(resposne)
        st.session_state.steps[str(len(msgs.messages) - 1)] = response["intermediate_steps"]