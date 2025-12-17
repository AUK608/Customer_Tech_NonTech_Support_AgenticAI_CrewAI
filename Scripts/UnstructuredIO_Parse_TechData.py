import os
import pytesseract

# keys for the services we will use

from dotenv import load_dotenv

###load environment variables
load_dotenv()

hf_token = os.getenv('HF_TOKEN')
mistralai_api = os.getenv('MISTRAL_API_KEY')
azure_token = os.getenv('AZURE_OPENAI_API_KEY')
api_version = str(os.getenv('AZURE_OPENAI_API_VERSION'))
azure_endpoint_url=os.getenv('AZURE_ENDPOINT_URL')
azure_deploy_url=os.getenv('AZURE_DEPLOYMENT_URL')


poppler_path = "C:\\Tasks\\GenAI_Tech_Report_Eriting_Tool_POC\\Py_3p11p0_TechReport_Env\\poppler_24p08p0\\Library\\bin"
os.environ["PATH"] += os.pathsep + poppler_path

# Configure PyTesseract for OCR
tes_path = "C:\\Tasks\\Multi_Agent_System_KYC\\Tesseract"
os.environ["PATH"] += os.pathsep + tes_path
pytesseract.pytesseract.tesseract_cmd = "C:\\Tasks\\Multi_Agent_System_KYC\\Tesseract\\tesseract.exe"

from unstructured.partition.pdf import partition_pdf

file_path = r"C:\Tasks\Customer_Tech_NonTech_Support_Agentic_RAG\CrewAI\Inputs\Dell_Guide.pdf"


# Reference: https://docs.unstructured.io/open-source/core-functionality/chunking
chunks = partition_pdf(
    filename=file_path,
    infer_table_structure=True,            # extract tables
    strategy="hi_res",                     # mandatory to infer tables

    extract_image_block_types=["Image"],   # Add 'Table' to list to extract image of tables
    # image_output_dir_path=output_path,   # if None, images and tables will saved in base64

    extract_image_block_to_payload=True,   # if true, will extract base64 for API usage

    chunking_strategy="by_title",          # or 'basic'
    max_characters=10000,                  # defaults to 500
    combine_text_under_n_chars=2000,       # defaults to 0
    new_after_n_chars=6000,

    #extract_images_in_pdf=True,          # deprecated
    #url=None,
)


from unstructured.partition.image import partition_image

img_path = r"C:\Tasks\Customer_Tech_NonTech_Support_Agentic_RAG\CrewAI\Inputs\Dell_Img.png"


# Reference: https://docs.unstructured.io/open-source/core-functionality/chunking
chunks_img = partition_image(
    filename=img_path,
    infer_table_structure=True,            # extract tables
    strategy="hi_res",                     # mandatory to infer tables

    extract_image_block_types=["Image"],   # Add 'Table' to list to extract image of tables
    # image_output_dir_path=output_path,   # if None, images and tables will saved in base64

    extract_image_block_to_payload=True,   # if true, will extract base64 for API usage

    chunking_strategy="by_title",          # or 'basic'
    max_characters=10000,                  # defaults to 500
    combine_text_under_n_chars=2000,       # defaults to 0
    new_after_n_chars=6000,

    #extract_images_in_pdf=True,          # deprecated
    #url=None,
)


import pandas as pd

xlsx_path = r"C:\Tasks\Customer_Tech_NonTech_Support_Agentic_RAG\CrewAI\Inputs\customer_ticket.csv"

table_html_list_1 = []

df = pd.read_csv(xlsx_path)

temp_html = df.to_html(index=False, border=False, escape=False, justify=None, classes=None)
#print(temp_html)
temp_html = temp_html.replace(' class=\"dataframe\"','').replace(' style=\"text-align: right;\"','').replace('\n','')#.replace(' ','')
table_html_list_1.append(temp_html)

print(table_html_list_1)
print(type(table_html_list_1))

'''from IPython.display import display, HTML
display(HTML(table_html_list_1[0]))'''


from bs4 import BeautifulSoup

# Original list with one HTML table string
#html_list = ['<table><thead><tr><th>id</th><th>name</th></tr></thead><tbody><tr><td>1</td><td>abc</td></tr><tr><td>2</td><td>def</td></tr></tbody></table>']
html_list = table_html_list_1

# Parse the HTML
soup = BeautifulSoup(html_list[0], 'html.parser')

# Extract the header and rows
thead = soup.find('thead')
rows = soup.find('tbody').find_all('tr')

# Create separate tables for each row
separate_tables = []
for row in rows:
    new_table = f"<table>{str(thead)}<tbody>{str(row)}</tbody></table>"
    separate_tables.append(new_table)

# Output
print(separate_tables)

from IPython.display import display, HTML
display(HTML(separate_tables[1]))

table_html_list = separate_tables


# separate tables from texts
tables = []
texts = []

for chunk in chunks:
    
    if "CompositeElement" in str(type(chunk)):
        texts.append(chunk)

        chunk_els = chunk.metadata.orig_elements
        
        for el in chunk_els:
            if "Table" in str(type(el)):
                tables.append(el.metadata.text_as_html)
        
import re

tables1 = [t for t in tables if 'Chapter 1:' not in t]

overall_tables = tables1 + table_html_list

# Get the images from the CompositeElement objects
def get_images_base64(chunks):
    images_b64 = []
    for chunk in chunks:
        if "CompositeElement" in str(type(chunk)):
            chunk_els = chunk.metadata.orig_elements
            for el in chunk_els:
                if "Image" in str(type(el)):
                    images_b64.append(el.metadata.image_base64)
    return images_b64

images = get_images_base64(chunks)
print(images)
for img in images:
    print(type(img))

images_img = get_images_base64(chunks_img)

overall_images = images + images_img

from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Prompt
prompt_text = """
You are an assistant tasked with summarizing tables and text.
Give a concise summary of the table or text.

Respond only with the summary, no additionnal comment.
Do not start your message by saying "Here is a summary" or anything like that.
Just give the summary as it is.

Table or text chunk: {element}

"""
prompt = ChatPromptTemplate.from_template(prompt_text)

# Summary chain
#model = ChatGroq(temperature=0.5, model="llama-3.1-8b-instant")
model = AzureChatOpenAI(
    azure_deployment="gpt4o",
    max_tokens=1024,
    temperature=0.1,
    api_version=api_version,
    azure_endpoint=azure_endpoint_url,
    api_key=os.getenv('AZURE_OPENAI_API_KEY'),
)
summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

# Summarize text
text_summaries = summarize_chain.batch(texts, {"max_concurrency": 3})

# Summarize tables
table_summaries = summarize_chain.batch(overall_tables, {"max_concurrency": 3})

#from langchain_openai import ChatOpenAI

prompt_template = """Describe the image in detail. For context,
                  the image is part of a screenshot of software windows, 
                  or process flows, or information regarding different 
                  software usages to help with customer support.
                  If image doesnt have any software or customer support related information and it contains only random or plain image then just mention as 'No Information Image'.
                  """
messages = [
    (
        "user",
        [
            {"type": "text", "text": prompt_template},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64,{image}"},
            },
        ],
    )
]

prompt = ChatPromptTemplate.from_messages(messages)

#chain = prompt | ChatOpenAI(model="gpt-4o-mini") | StrOutputParser()
chain = prompt | model | StrOutputParser()


image_summaries = chain.batch(overall_images)

img_sum = [i for i in image_summaries if 'No Information Image' not in i]
overall_img = [overall_images[i] for i in range(len(image_summaries)) if 'No Information Image' not in image_summaries[i]]

import uuid
from langchain.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
#from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_openai import AzureOpenAIEmbeddings
import pickle

embeddings = AzureOpenAIEmbeddings(
    model="text-embedding-ada-002",
    api_version=api_version,
    azure_endpoint=azure_endpoint_url,
    api_key=os.getenv('AZURE_OPENAI_API_KEY'),
)

persist_embeddings = r"C:\Tasks\Customer_Tech_NonTech_Support_Agentic_RAG\CrewAI\Scripts\ChromaRetriever_DocstorePickle_Save"
pkl_save_docstore = r"C:\Tasks\Customer_Tech_NonTech_Support_Agentic_RAG\CrewAI\Scripts\ChromaRetriever_DocstorePickle_Save\docstore_pickle_save_docstore.pkl"

per_vectorstore = Chroma(collection_name="per_multi_modal_rag", embedding_function=embeddings, persist_directory=persist_embeddings)

# The storage layer for the parent documents
per_store = InMemoryStore()
per_id_key = "doc_id"

# The retriever (empty to start)
per_retriever = MultiVectorRetriever(
    vectorstore=per_vectorstore,
    docstore=per_store,
    id_key=per_id_key,
)

# Add texts
doc_ids = [str(uuid.uuid4()) for _ in texts]
summary_texts = [
    Document(page_content=summary, metadata={per_id_key: doc_ids[i]}) for i, summary in enumerate(text_summaries)
]
per_retriever.vectorstore.add_documents(summary_texts)
per_retriever.docstore.mset(list(zip(doc_ids, texts)))

# Add tables
table_ids = [str(uuid.uuid4()) for _ in overall_tables]
summary_tables = [
    Document(page_content=summary, metadata={per_id_key: table_ids[i]}) for i, summary in enumerate(table_summaries)
]
per_retriever.vectorstore.add_documents(summary_tables)
per_retriever.docstore.mset(list(zip(table_ids, overall_tables)))


# Add image summaries
img_ids = [str(uuid.uuid4()) for _ in overall_img]
print("\n-----img_ids = = = =  = \n", img_ids)
summary_img = [
    Document(page_content=summary, metadata={per_id_key: img_ids[i]}) for i, summary in enumerate(img_sum)
]
print("\n-----summary_img = = = =  = \n", summary_img)
per_retriever.vectorstore.add_documents(summary_img)
per_retriever.docstore.mset(list(zip(img_ids, overall_img)))
print("\n------ret get rel doc = = = = \n",per_retriever.get_relevant_documents)
print("\n------ret get name = = = = \n",per_retriever.get_name)

per_vectorstore.persist()

with open(pkl_save_docstore, "wb") as f:
    pickle.dump(per_store, f)