import os
import yaml

from PyPDF2 import PdfReader

from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

from langchain.agents import create_csv_agent
from langchain.llms import OpenAI

###################################################################################
# Set up
###################################################################################

##############################################
# Set up APIs
##############################################
with open('temp/local_apis.yaml', 'r') as file:
    apis = yaml.safe_load(file)

os.environ['OPENAI_API_KEY'] = apis['openai']
os.environ['HUGGINGFACEHUB_API_TOKEN'] = apis['huggingface']

##############################################
# Set up models
##############################################
openai_model_name = 'text-davinci-003'
hf_embed_name = 'sentence-transformers/all-mpnet-base-v2'
model_kwargs = {'device': 'cuda'}

##############################################
# Get files
##############################################
doc_reader = PdfReader('temp/WCooper CV.pdf')

###################################################################################
# Get some text
###################################################################################

# PDF Reader
# read data from the file and put them into a variable called raw_text
raw_text = ''
for i, page in enumerate(doc_reader.pages):
    text = page.extract_text()
    if text:
        raw_text += text

raw_text_clean = raw_text.replace("•", "\n")


# Splitting up the text into smaller chunks for indexing
text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 200,
    chunk_overlap  = 50, #striding over the text
    length_function = len,
)
texts = text_splitter.split_text(raw_text_clean)
print(len(texts))

##############################################
# Get Embeddings
##############################################

# Download embeddings from OpenAI
embeddings = OpenAIEmbeddings()
# embeddings = HuggingFaceEmbeddings(model_name=hf_embed_name, model_kwargs=model_kwargs)
docsearch = FAISS.from_texts(texts, embeddings)

docsearch.embedding_function

query = "Does wade know python?"
docs = docsearch.similarity_search(query)

for doc in docs:
    print(doc)

##############################################
# Run Q&A
##############################################


openai_model = OpenAI(model_name=openai_model_name)

chain = load_qa_chain(llm=openai_model,
                      chain_type="stuff") # we are going to stuff all the docs in at once


query = "Does wade know python?"
docs = docsearch.similarity_search(query)


api_usage = {}
with get_openai_callback() as cb:
    response = chain.run(input_documents=docs, question=query)
    api_usage['total_tokens'] = cb.total_tokens
    api_usage['prompt_tokens'] = cb.prompt_tokens
    api_usage['completion_tokens'] = cb.completion_tokens
    api_usage['total_cost'] = cb.total_cost

print(response)
print(api_usage)

