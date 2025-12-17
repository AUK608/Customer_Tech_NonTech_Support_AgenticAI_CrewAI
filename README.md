# Customer_Tech_NonTech_Support_AgenticAI_CrewAI
Streamlit based Chatbot for Tech and Non-Tech Customer Queries using CrewAI framework to showcase RAG based AgenticAI in python v3.11.0

- This Repo includes sample inputs/PDFs used, requirements.txt, screenshot of the chatbot, scripts
- Script uses Azure embeddings and LLM(gpt4o)
- Install packages as "python.exe -m pip install -r Py3p11p0_CustSupport_CrewAI_requirements.txt"
- I'm using unstructured.io for parsing the Tech PDF and Images(process is shown in UnstructuredIO_Parse_TechData.py), and Non-Tech PDF using CrewAI Tools, also unstructured.io requires tesseract-ocr and poppler
- Storing parsed data in ChromaDB and using Langchain's MultiVectorRetriever
- Modify the paths as required from the scripts
- Run chatbot script as "python.exe -m streamlit run Customer_Tech_NonTech_Support_AgenticRAG_CrewAI.py"
