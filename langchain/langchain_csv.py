import os
import yaml
import pandas as pd

from langchain.agents import create_csv_agent
from langchain.llms import OpenAI

###################################################################################
# Set up os env vars
###################################################################################
with open('temp/local_apis.yaml', 'r') as file:
    apis = yaml.safe_load(file)

os.environ['OPENAI_API_KEY'] = apis['openai']
os.environ['HUGGINGFACEHUB_API_TOKEN'] = apis['huggingface']

openai_model_name = 'text-davinci-003'


###################################################################################
# Test CSV data store
###################################################################################

##############################################
# Setting it up
##############################################

# Get a sample file
os.chdir(f'{os.getcwd()}/temp/')
print(os.getcwd())
os.system('wget -q https://www.dropbox.com/s/8y6a1zloiscuo5d/black_friday_sales.zip')
os.system('unzip -q black_friday_sales.zip')
df = pd.read_csv('train.csv')


agent = create_csv_agent(OpenAI(temperature=0),
                         'train.csv',
                         verbose=True)

agent.agent.llm_chain.prompt.template

agent.run("how many rows are there?")

agent.run("how many people are female?")

agent.run("how many people have stayed more than 3 years in the city?")

agent.run("how many people have stayed more than 3 years in the city and are female?")

agent.run("Are there more males or females?")
