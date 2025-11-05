import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler  # this allows to communicate all these tools in themselves
import os
from dotenv import load_dotenv

# Arxiv and Wikipedia Tools
api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
arxiv_tool = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

search = DuckDuckGoSearchRun(name="Search")

st.title("🔍 LangChain - Chat With Search")
"""
In this example, we're using `StreamlitCallbackHandler` to display the thoughts and actions of the agent in an interactive Streamlit App.
Try more LangChain Streamlit Agents examples at [github.com/langchain-ai/streamlit-agent]
"""

# Sidebar for settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq Api Key:", type="password")

# creating session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I am a Chatbot who can search the Web. How Can I help you...?"}
    ]

#Display old messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="Type Your Question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.1-8b-instant", streaming=True)
    tools = [search, arxiv_tool, wiki_tool]

    # make agent to invoke those tools
    search_agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True
    )

#Generate assistant’s reply
    with st.chat_message("assistant"): #when assistance is giving message
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
        response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
        #by getting response handling it into it
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)

#Runs the agent with the conversation history.

"""Gets the assistant’s reply.
Saves it into history.

Displays it in the chat."""
