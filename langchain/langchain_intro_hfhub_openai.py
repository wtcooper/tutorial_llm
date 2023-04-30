import os
import yaml
import pandas as pd

import tiktoken
from langchain.agents import create_csv_agent
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain import HuggingFaceHub, LLMChain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

###################################################################################
# Set up os env vars
###################################################################################

with open('temp/local_apis.yaml', 'r') as file:
    apis = yaml.safe_load(file)

os.environ['OPENAI_API_KEY'] = apis['openai']
os.environ['HUGGINGFACEHUB_API_TOKEN'] = apis['huggingface']


# openai_model_name = 'text-davinci-003'
openai_model_name = 'text-babbage-001'
hf_model_name = 'google/flan-t5-large'


###################################################################################
# Set up prompts
###################################################################################

template = """Question: {question}

Answer: Let's think step by step. """

prompt = PromptTemplate(template=template, input_variables=["question"])

# user question
question = "Which NFL team won the Super Bowl in the 2010 season?"



##############################################
# Basic HF Hub model
##############################################

# initialize Hub LLM
hf_model = HuggingFaceHub(
    repo_id=hf_model_name,
    model_kwargs={"temperature":0, "max_length":64}
)

# create prompt template > LLM chain
hub_llm_chain = LLMChain(
    prompt=prompt,
    llm=hf_model
)

# ask the user question about NFL 2010
print(hub_llm_chain.run(question))

# Multiple questions
qs = [
    {'question': "Which NFL team won the Super Bowl in the 2010 season?"},
    {'question': "If I am 6 ft 4 inches, how tall am I in centimeters?"},
    {'question': "Who was the 12th person on the moon?"},
    {'question': "How many eyes does a blade of grass have?"}
]
res = hub_llm_chain.generate(qs)
print(res)



##############################################
# Basic OpenAI model with cost
##############################################

openai_model = OpenAI(model_name=openai_model_name)

openai_llm_chain = LLMChain(
    prompt=prompt,
    llm=openai_model
)

# store the usage information from callback
api_usage = {}

with get_openai_callback() as cb:
    response = openai_llm_chain.run(question)
    api_usage['total_tokens'] = cb.total_tokens
    api_usage['prompt_tokens'] = cb.prompt_tokens
    api_usage['completion_tokens'] = cb.completion_tokens
    api_usage['total_cost'] = cb.total_cost

print(response)
print(api_usage)

# manually count tokens - same
enc = tiktoken.encoding_for_model(openai_model_name)
input_text = enc.encode(prompt.format(question=question))
output_text = enc.encode(response)


token_total = len(input_text + output_text)
