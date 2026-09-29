from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from typing import Annotated

class ChatState(BaseModel):
    messages:Annotated[list,add_messages]##add_message append coming message into messages as a list
llm=ChatGoogleGenerativeAI(model="gemini-3.6-flash")
def chatBotNode(state:ChatState) -> ChatState:
    res=llm.invoke(state.messages)
    state.messages=[res]
    return state
memory=InMemorySaver()##needs to pass the config with messages
grapgh=StateGraph(ChatState)
grapgh.add_node("chatBotNode",chatBotNode)
grapgh.add_edge(START,"chatBotNode")
grapgh.add_edge("chatBotNode",END)

grapgh=grapgh.compile(checkpointer=memory)
config={"configurable":{"thread_id":"My-bot-1"}}


while True:
    query=input("User:")
    if query.lower() in ["quit","exit"]:
        print("Thanks for the Visit")
        break

    res=grapgh.invoke({"messages":[{"role":"user","content":query}]},config)
    print("Ai:",res["messages"][-1].text)