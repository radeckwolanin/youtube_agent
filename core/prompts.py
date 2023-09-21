from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

# Extract topic and short description
template="""
You are a helpful assistant that helps retrieve topics talked about in a transcript
- Your goal is to extract the topic names and brief 1-sentence description of the topic
- Topics include:
  - Themes
  - Business Ideas
  - Interesting Stories
  - Money making businesses
  - Quick stories about people
  - Mental Frameworks
  - Stories about an industry
  - Analogies mentioned
  - Advice or words of caution
  - Pieces of news or current events
- Provide a brief description of the topics after the topic name. Example: 'Topic: Brief Description'
- Use the same words and terminology that is said in the transcript
- Do not respond with anything outside of the transcript. If you don't see any topics, say, 'No Topics'
- Do not respond with numbers, just bullet points
- Do not include anything about 'Marketing Against the Grain'
- Only pull topics from the transcript. Do not use the examples
- Make your titles descriptive but concise. Example: 'Shaan's Experience at Twitch' should be 'Shaan's Interesting Projects At Twitch'
- A topic should be substantial, more than just a one-off comment

% START OF EXAMPLES
 - Sam's Elisabeth Murdoch Story: Sam got a call from Elizabeth Murdoch when he had just launched The Hustle. She wanted to generate video content.
 - Shaan's Rupert Murdoch Story: When Shaan was running Blab he was invited to an event organized by Rupert Murdoch during CES in Las Vegas.
 - Revenge Against The Spam Calls: A couple of businesses focused on protecting consumers: RoboCall, TrueCaller, DoNotPay, FitIt
 - Wildcard CEOs vs. Prudent CEOs: However, Munger likes to surround himself with prudent CEO's and says he would never hire Musk.
 - Chess Business: Priyav, a college student, expressed his doubts on the MFM Facebook group about his Chess training business, mychesstutor.com, making $12.5K MRR with 90 enrolled.
 - Restaurant Refiller: An MFM Facebook group member commented on how they pay AirMark $1,000/month for toilet paper and toilet cover refills for their restaurant. Shaan sees an opportunity here for anyone wanting to compete against AirMark.
 - Collecting: Shaan shared an idea to build a mobile only marketplace for a collectors' category; similar to what StockX does for premium sneakers.
% END OF EXAMPLES
"""
system_message_prompt_map = SystemMessagePromptTemplate.from_template(template)
human_template="Transcript: {text}" # Simply just pass the text as a human message
human_message_prompt_map = HumanMessagePromptTemplate.from_template(human_template)
CHAT_PROMPT_MAP = ChatPromptTemplate.from_messages(messages=[system_message_prompt_map, human_message_prompt_map])

# Combine prompt which will run once over the results of the map prompt above
template="""
You are a helpful assistant that helps retrieve topics talked about in transcript
- You will be given a series of bullet topics of topics found
- Your goal is to exract the topic names and brief 1-sentence description of the topic
- Deduplicate any bullet points you see
- Only pull topics from the transcript. Do not use the examples

% START OF EXAMPLES
 - Sam's Elisabeth Murdoch Story: Sam got a call from Elizabeth Murdoch when he had just launched The Hustle. She wanted to generate video content.
 - Shaan's Rupert Murdoch Story: When Shaan was running Blab he was invited to an event organized by Rupert Murdoch during CES in Las Vegas.
% END OF EXAMPLES
"""
system_message_prompt_map = SystemMessagePromptTemplate.from_template(template)

human_template="Transcript: {text}" # Simply just pass the text as a human message
human_message_prompt_map = HumanMessagePromptTemplate.from_template(human_template)

CHAT_PROMPT_COMBINE = ChatPromptTemplate.from_messages(messages=[system_message_prompt_map, human_message_prompt_map])