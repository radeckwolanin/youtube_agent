import os
import json
import pandas as pd
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
from core.tools import CustomYTSearchTool, CustomYTTranscribeTool, SummarizationTool, VectorDBCheckStatus

load_dotenv() # Load environment variables from .env file

st.set_page_config(
    page_title="YouTube Agent", 
    page_icon="📺",
    layout="centered", # wide
    initial_sidebar_state="expanded",
    menu_items={ # TODO
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

st.title("📺 YouTube Agent")

msgs = StreamlitChatMessageHistory()

memory = ConversationBufferMemory(
    chat_memory=msgs,
    k=5,
    return_messages=True, 
    memory_key="chat_history", 
    output_key="output"
)

if len(msgs.messages) == 0: #or st.sidebar.button("Reset chat history"):
    msgs.clear()
    msgs.add_ai_message("As your YouTube agent, I can perform many task related to YT videos. For example you can start by searching any subject like 'search todays top global news' or 'transcribe this video'.")
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
    
    tools = []
    tools.append(CustomYTSearchTool())
    tools.append(CustomYTTranscribeTool())
    tools.append(SummarizationTool())
    tools.append(VectorDBCheckStatus())
    
    chat_agent = ConversationalChatAgent.from_llm_and_tools(llm=llm, tools=tools)
    #chat_agent = initialize_agent(
    #    tools, 
    #    llm, 
    #    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    #    verbose=False,
    #    max_iterations=5,
    #    memory=memory,
    #)
    
    executor = AgentExecutor.from_agent_and_tools(
        agent=chat_agent,
        tools=tools,
        memory=memory,
        return_intermediate_steps=True,
        handle_parsing_errors=True
    )
    
    
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        
        response = executor(prompt, callbacks=[st_cb])
        
        st.write(response["output"]) # if executor 
        #st.write(response) # if initialize agent
        #print(response)
        #print("\n\n")
        
        # To print json returned from CustomYTSearch as table
        #for inter_step in response["intermediate_steps"]:
        #    if inter_step[0].tool == "CustomYTSearch":
        #        data = json.loads(inter_step[1])
        #        df = pd.DataFrame(data['videos'])
        #        st.table(df)
                
        st.session_state.steps[str(len(msgs.messages) - 1)] = response["intermediate_steps"]
        