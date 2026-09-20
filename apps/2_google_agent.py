from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from context.dev import ContextDev
import os


load_dotenv()
client = ContextDev()
llm=ChatGoogleGenerativeAI(model="gemini-3.6-flash")
agent = create_agent(
    model=llm,
    tools=[client.web.web_scrape_md],
    system_prompt="Your an agent and search for any question on google"

)

while True:
    query=input("User:")
    if query.lower()== ["quit","exit","bye"]:
        print("Good Bye")
        break
    response = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": query
            }
        ]
    })
    print("Ai:",response["messages"][-1].text)

