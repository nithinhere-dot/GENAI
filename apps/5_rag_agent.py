from dotenv import load_dotenv
gemini_api_key=load_dotenv()
import os

print(os.getenv("GOOGLE_API_KEY"))

from langchain_community.document_loaders import PyPDFLoader,PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI
from langchain_community.vectorstores import InMemoryVectorStore
import os
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

import streamlit as st

## data in st session
if "document_Uploaded" not in st.session_state:
    st.session_state.document_uploaded=False

if "agent" not in st.session_state:
    st.session_state.agent=None

if "vector_store" not in st.session_state:
    st.session_state.vector_store=None

if "messages" not in st.session_state:
    st.session_state.messages=[]



def process_document(path):

    ## Load the Documents
    print(gemini_api_key)
    loader=PyPDFDirectoryLoader(path)
    docs=loader.load()

    ##split into multiple chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    docs=splitter.split_documents(documents=docs)
    print("Number of chunks:", len(docs))

    ##embediings and vector DB
    embeddings=GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")
    vector_db=InMemoryVectorStore.from_documents(
        documents=docs,
        embedding=embeddings
    )
    

    ##create a agent-tool,llm,prompt
    llm=ChatGoogleGenerativeAI(model="gemini-3.7-flash")

    @tool
    def retriver_tool(query:str):
        """
            retrive documents relevent to a query from the knowledge base
        """
        docs=vector_db.similarity_search(query=query,k=3)
        context=""
        for doc in docs:
            context+=doc.page_content+"\n\n"
        return context
    system_prompt="""you are a helpful assistent that answers question using retrived context My Knowledge base consists of the details from the uploaded document.Always use the 'retriver_tool' tool for questions requiring external knowledge"""

    memory=InMemorySaver()
    agent=create_agent(
        model=llm,
        tools=[retriver_tool],
        system_prompt=system_prompt,
        checkpointer=memory
    )
    st.session_state.agent=agent
    st.session_state.document_uploaded=True

##upload ui
if not st.session_state.document_uploaded:
    uploaded =st.file_uploader(label="Select PDF files",type=["pdf"],accept_multiple_files=True)
    if uploaded:
        with st.spinner("Processing"):
            path="./doc_files/"
            os.makedirs(path, exist_ok=True)

            for file in uploaded:
                with open(path+file.name,"wb") as f:
                    f.write(file.getvalue())
            
            process_document(path)
            st.rerun()

##chat ui
if st.session_state.document_uploaded and st.session_state.agent:

    for message in st.session_state.messages:
        role=message.get("role")
        content=message.get("content")
        st.chat_message(role).markdown(content)
    
    query=st.chat_input("Ask anything releted to uploaded documents....")
    if query:
        st.session_state.messages.append({"role":"user","content":query})
        st.chat_message("user").markdown(query)
        res=st.session_state.agent.invoke(
            {"messages":[{"role":"user","content":query}]},
            {"configurable":{"thread_id":1}}
        )

        answer=res["messages"][-1].text
        st.session_state.messages.append({"role":"ai","content":answer})
        st.chat_message("ai").markdown(answer)