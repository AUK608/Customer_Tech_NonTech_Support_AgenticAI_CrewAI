import os
from base64 import b64decode
import uuid
from langchain.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
#from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import AzureChatOpenAI
from crewai.tools import BaseTool, tool
from crewai import Crew, Task, Agent, LLM
from crewai_tools import PDFSearchTool
import traceback
import pickle
from dotenv import load_dotenv

###load environment variables
load_dotenv()

azure_token = os.getenv('AZURE_OPENAI_API_KEY')
api_version = str(os.getenv('AZURE_OPENAI_API_VERSION'))
azure_endpoint_url=os.getenv('AZURE_ENDPOINT_URL')
azure_deploy_url=os.getenv('AZURE_DEPLOYMENT_URL')

persist_embeddings_2 = r"C:\Tasks\Customer_Tech_NonTech_Support_Agentic_RAG\CrewAI\Scripts\ChromaRetriever_DocstorePickle_Save"
pkl_save_docstore_2 = r"C:\Tasks\Customer_Tech_NonTech_Support_Agentic_RAG\CrewAI\Scripts\ChromaRetriever_DocstorePickle_Save\docstore_pickle_save_docstore.pkl"

embeddings = AzureOpenAIEmbeddings(
    model="text-embedding-ada-002",
    api_version=api_version,
    azure_endpoint=azure_endpoint_url,
    api_key=azure_token,
)

per_vectorstore_2 = Chroma(collection_name="per_multi_modal_rag", embedding_function=embeddings, persist_directory=persist_embeddings_2)

# The storage layer for the parent documents
#per_store_1 = InMemoryStore()
with open(pkl_save_docstore_2, "rb") as f:
    per_store_2 = pickle.load(f)

per_id_key_2 = "doc_id"

# The retriever (empty to start)
per_retriever_2 = MultiVectorRetriever(
    vectorstore=per_vectorstore_2,
    docstore=per_store_2,
    id_key=per_id_key_2,
)

model = AzureChatOpenAI(
    azure_deployment="gpt4o",
    max_tokens=1024,
    temperature=0.1,
    api_version=api_version,
    azure_endpoint=azure_endpoint_url,
    api_key=azure_token,
)

llm = LLM(
    model="azure/gpt4o",
    base_url=azure_deploy_url,
    api_key=azure_token
)

nontech_rag_tool = PDFSearchTool(
    pdf=r"C:\Tasks\Customer_Tech_NonTech_Support_Agentic_RAG\CrewAI\Inputs\Company_NonTech_FAQ.pdf",
    config=dict(
        llm=dict(
            provider="azure_openai",
            config=dict(
                model="gpt4o",
                ),
            ),
        embedder=dict(
                provider="huggingface",
                config=dict(
                    model="BAAI/bge-small-en-v1.5",
                ),
            ),
        )
    )

@tool("issue router tool")
def issue_router_tool(question:str):
    """Router Function"""
    prompt = f"""Based on the Question provide below determine the following:
    1. Is the question directed at Technical Issue or Query ?
    2. Is the question directed at Non-Technical Issue or Query ?
    Question: {question}

    RESPONSE INSTRUCTIONS:
    - Answer either 1 or 2.
    - Answer should strictly be a string.
    - Do not provide any preamble or explanations except for 1 or 2.

    OUTPUT FORMAT:
    1
    """
    response = model.invoke(prompt).content
    if response == "1":
        return 'TechnicalIssue'
    else:
        return 'NonTechnicalIssue'
    
def multimodal_context_retriever(caller, question:str):
    final_context_list = []
    context_list = per_retriever_2.invoke(question)

    if caller == "agent_tool":
        return context_list
    else:
        for inum in range(len(context_list)):
            doc = context_list[inum]
            
            try:
                if '<table>' in doc and '<tr>' in doc:
                    final_context_list.append(doc)
                else:
                    b64decode(doc)
                    final_context_list.append(doc) 
            except Exception as e:
                continue
        
        return final_context_list

def multimodal_summary_retriever(question:str):
    final_summary_list = []
    summary_list = []

    context_list = multimodal_context_retriever("agent_tool", question)
    
    summary_list_temp = per_retriever_2.vectorstore.similarity_search(question)
    for doc in summary_list_temp:
        summary_list.append(doc.page_content)
    
    for inum in range(len(context_list)):
        doc = context_list[inum]

        try:
            if '<table>' in doc and '<tr>' in doc:
                final_summary_list.append(doc)
            else:
                b64decode(doc)
                if inum < len(summary_list):
                    final_summary_list.append(summary_list[inum]) 
        except Exception as e:
            final_summary_list.append(doc.text)
    
    return final_summary_list

def nontech_retriever(question:str):
    return nontech_rag_tool.run(question)

@tool("retriever tool")
def retriever_tool(router_resposne:str, question:str):
    """Retriever Function"""
    if router_resposne == "TechnicalIssue":
        return multimodal_summary_retriever(question)
    else:
        return nontech_retriever(question)

Issue_Intake_Coordinator_Agent = Agent(
  role='Issue_Type_Router',
  goal='Based on User Question, Route to either a Technical Issue Retriever or a Non-Technical Issue Retriever',
  backstory=(
    "You are an expert at routing a user question to a Technical Issue Retriever or a Non-Technical Issue Retriever."
    "If routed to Technical Issue Retriever and doesnt get any solution then route to Non-Technical Issue Retriever and vice versa."
  ),
  verbose=True,
  allow_delegation=False,
  llm=llm,
)

Issue_Intake_Coordinator_Task = Task(
    description=(
        "Understand the question {question} and Route to either a Technical Issue Retriever or a Non-Technical Issue Retriever"
        "Return a single word 'TechnicalIssue' if question is related to Technical Issue."
        "Return a single word 'NonTechnicalIssue' if question is related to Non-Technical Issue."
        "If routed to Technical Issue Retriever and doesnt get any solution then route to Non-Technical Issue Retriever and vice versa."
        "Do not provide any other explaination. Use the tool given as it is."
      ),
    expected_output=("Based on tool given, just give 'TechnicalIssue' or 'NonTechnicalIssue' as output and nothing else."),
    agent=Issue_Intake_Coordinator_Agent,
    tools=[issue_router_tool]
)

Knowledge_Retrieval_Specialist_Agent = Agent(
  role='Knowledge_Retrieval_Specialist',
  goal='Based on User Question, Retrieve Related/Corresponding Information from either Technical Issue Retriever or Non-Technical Issue Retriever',
  backstory=(
    "You are an expert at retrieving information related/corresponding from Technical Issue Retriever or a Non-Technical Issue Retriever based on User Question."
    "If no related/corresponding information/solution found from Technical Issue Retriever then get related/corresponding information/solution from Non-Technical Issue Retriever and vice versa."
  ),
  verbose=True,
  allow_delegation=False,
  llm=llm,
)

Knowledge_Retrieval_Specialist_Task = Task(
    description=(
        "Understand the User Question {question} and Retrieve Related/Corresponding Information from either Technical Issue Retriever or Non-Technical Issue Retriever based on the route provided/decided by previous Issue_Intake_Coordinator Agent and its Task"
        "Use the provided tool only and return the tool output as it is and do not provide any other explaination."
        "If no related/corresponding information/solution found from Technical Issue Retriever then get related/corresponding information/solution from Non-Technical Issue Retriever and vice versa."
      ),
    expected_output=("Based on tool given, just give Technical Issue Retriever's or Non-Technical Issue Retriever's output as it is and nothing else."),
    agent=Knowledge_Retrieval_Specialist_Agent,
    context=[Issue_Intake_Coordinator_Task],
    tools=[retriever_tool]
)

Resolution_Synthesizer_Agent = Agent(
  role='Resolution_Synthesizer',
  goal='Based on User Question and Retrieved Content/Information from Previous Knowledge_Retrieval_Specialist Agent and Task, Analyze and Generate Step by Step Resolution',
  backstory=(
    "Act as a Customer Support Assistant as in You are an expert at Analyzing and Generating Step by Step Resolution based on Content/Information which can have Image's, HTML Table's, Text's Data Retrieved from Previous Knowledge_Retrieval_Specialist Agent and Task which can be either Technical Issue or Non-Technical Issue based on User Question."
  ),
  verbose=True,
  allow_delegation=False,
  llm=llm,
)

Resolution_Synthesizer_Task = Task(
    description=(
        "Understand/Analyze the Retrieved Content/Information which can have Connected or Sequence of Continued Image's, HTML Table's, Text's Data from Previous Knowledge_Retrieval_Specialist Agent and Task and then Generate Step by Step Resolution for a Customer Support Assistant Role based on User Question"
      ),
    expected_output=("Step by Step Resolution based on Retrieved Content/Information"),
    agent=Resolution_Synthesizer_Agent,
    context=[Knowledge_Retrieval_Specialist_Task]
)

QA_Analyst_Agent = Agent(
  role='QA_Analyst',
  goal='Based on the Step by Step Resolution received from Previous Resolution_Synthesizer Agent and Task, to Validate Technical Accuracy, Check for Contradiction, Ensure Clarity',
  backstory=(
    "Act as a QA Analyst and Validate Technical Accuracy, Check for Contradiction, Ensure Clarity based on Step by Step Resolution received from Resolution_Synthesizer Agent and Task."
    "If everything is Correctly Analyzed without any Corrections required then keep the Resolution as it is, else Correct/Modify the Step by Step Resolution based on User Question and Recieved Content/Information."
  ),
  verbose=True,
  allow_delegation=False,
  llm=llm,
)

QA_Analyst_Task = Task(
    description=(
        "As a QA Analyst, Understand/Analyze the Step by Step Resolution received from Resolution_Synthesizer Agent and Task, then Validate Technical Accuracy, Check for Contradiction, Ensure Clarity."
        "If any Corrections/Modifications are Required, then Correct/Modify it or else keep the Step by Step Resolution as it is."
      ),
    expected_output=("Step by Step Resolution based on Validated Technical Accuracy, Contradiction Checks, Clarity Ensurance"),
    agent=QA_Analyst_Agent,
    context=[Resolution_Synthesizer_Task]
)

Tech_Support_Assistant_Crew = Crew(
                agents=[Issue_Intake_Coordinator_Agent, Knowledge_Retrieval_Specialist_Agent, Resolution_Synthesizer_Agent, QA_Analyst_Agent],
                tasks=[Issue_Intake_Coordinator_Task, Knowledge_Retrieval_Specialist_Task, Resolution_Synthesizer_Task, QA_Analyst_Task],
                verbose=True,
            )

import streamlit as st

st.title("AI Technical Support Assistant")
st.markdown(
    "AI Assistant Which Can Support For Customer Queries Related To Technical And Non-Technical"
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message.get("is_html"):
            st.markdown(message["content"], unsafe_allow_html=True)
        else:
            st.markdown(message["content"])

if prompt := st.chat_input("Ask Technical or Non-Technical Queries"):
    all_response_html = ""

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    crew_response = Tech_Support_Assistant_Crew.kickoff(inputs={"question": prompt})
    all_response_html += f"{crew_response}<br>"

    if str(Issue_Intake_Coordinator_Task.output) == 'TechnicalIssue':
        retriever_response = multimodal_context_retriever("UI", prompt)

        all_response_html += "<br>" + "Below is/are Table(s) and Image(s) for Reference :"
        for inum in range(len(retriever_response)):
            
            doc = retriever_response[inum]
            try:
                if '<table>' in doc and '<tr>' in doc:
                    all_response_html += "<p></p>"
                    all_response_html += doc
                else:
                    b64decode(doc)
                    all_response_html += "<p></p>"
                    all_response_html += f'<img src="data:image/png;base64,{doc}" alt="Image">'
            except Exception as e:
                print("\n exception in streamlit = = =  = \n", str(traceback.format_exc()))
                continue

    with st.chat_message("assistant"):
        st.markdown(all_response_html, unsafe_allow_html=True)
    
    st.session_state.messages.append({"role": "assistant", "content": all_response_html, "is_html": True})