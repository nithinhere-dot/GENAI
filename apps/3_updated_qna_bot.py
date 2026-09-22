from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from context.dev import ContextDev
from langgraph.checkpoint.memory import InMemorySaver
import streamlit as st
import os


load_dotenv()
client = ContextDev()

llm=ChatGoogleGenerativeAI(model="gemini-3.6-flash",Streaming=True)
st.title("Complete QnA bot")
st.markdown("My QnA Bot with Langchain,google gemini and updated memory!")


if "memory" not in st.session_state:
    st.session_state.memory=InMemorySaver()
    st.session_state.messages=[] 

agent = create_agent(
    model=llm,
    tools=[client.web.web_scrape_md],
    system_prompt="Your an agent and search for any question on google",
    checkpointer=st.session_state.memory

)


for message in st.session_state.messages:
    role=message["role"]
    content=message["content"]
    st.chat_message(role).markdown(content)


query=st.chat_input("Ask Anything:")
if query:
    st.chat_message("user").markdown(query)
    st.session_state.messages.append({"role":"user","content":query})
    response = agent.stream({
        "messages": [
            {
                "role": "user",
                "content": query
            }
        ]
    },{"configurable":{"thread_id":"1"}},
    stream_mode="messages")
    ai_container=st.chat_message("ai")
    with ai_container:
        space=st.empty()

        message=""

        for chunk in response:
            message=message+chunk[0].text
            space.write(message)
        st.session_state.messages.append({"role":"ai","content":message})

