import os
import yaml

from langchain import PromptTemplate, LLMChain, HuggingFaceHub, HuggingFacePipeline
from langchain.chains import SimpleSequentialChain

from transformers import AutoTokenizer, pipeline, AutoModelForSeq2SeqLM
from urllib.request import urlretrieve


##############################################
# Set up os env vars
##############################################
with open('temp/local_apis.yaml', 'r') as file:
    apis = yaml.safe_load(file)

os.environ['HUGGINGFACEHUB_API_TOKEN'] = apis['huggingface']

model_id = "google/flan-t5-large"

##############################################
# Set up prompt
##############################################

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])


##############################################
# Load Huggingface from Hub (need API)
##############################################

hub_llm = HuggingFaceHub(repo_id="google/flan-t5-large",
                                        model_kwargs={"temperature":0,
                                                      "max_length":64})
hub_llm_chain = LLMChain(prompt=prompt, llm=hub_llm)


question = "What is the capital of France?"
print(hub_llm_chain.run(question))


##############################################
# Load Huggingface models locally - langchain pipeline
##############################################

local_llm = HuggingFacePipeline.from_model_id(
    model_id=model_id,
    task="text2text-generation",
    device=0
    )

local_llm_chain = LLMChain(prompt=prompt, llm=local_llm)

question = "What is the capital of France?"
print(local_llm_chain.run(question))


##############################################
# Manual download and create pipeline
##############################################

# model_id = 'chavinlo/alpaca-native'
model_id = 'google/flan-t5-large'# go for a smaller model if you dont have the VRAM
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id, load_in_8bit=True, device_map='auto')

pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=100
)

local_hf_llm = HuggingFacePipeline(pipeline=pipe)

local_hf_llm_chain = LLMChain(prompt=prompt, llm=local_hf_llm)

question = "What is the capital of France?"
print(local_hf_llm_chain.run(question))



