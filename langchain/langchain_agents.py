import os
import yaml
from bs4 import BeautifulSoup
import requests

from langchain.agents import Tool
from langchain.tools import BaseTool

from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

from langchain.utilities import WikipediaAPIWrapper
from langchain.utilities import PythonREPL
from langchain.tools import DuckDuckGoSearchRun
from langchain.agents import initialize_agent

###################################################################################
# Set up os env vars
###################################################################################
with open('temp/local_apis.yaml', 'r') as file:
    apis = yaml.safe_load(file)

os.environ['OPENAI_API_KEY'] = apis['openai']
os.environ['HUGGINGFACEHUB_API_TOKEN'] = apis['huggingface']


###################################################################################
# Canned Tools: Wikipedia DuckDuckGo PythonRepl
###################################################################################

##############################################
# Setting it up
##############################################

"""## Setting up wikipedia"""

wikipedia = WikipediaAPIWrapper()
wikipedia.run('Langchain')

"""## REPL"""
python_repl = PythonREPL()
python_repl.run("print(17*2)")


"""## Duck Duck Go"""
search = DuckDuckGoSearchRun()
search.run("Tesla stock price?")


##############################################
# Putting them together
##############################################

llm = OpenAI(temperature=0)

tools = [
    Tool(
        name = "python repl",
        func=python_repl.run,
        description="useful for when you need to use python to answer a question. You should input python code"
    )
]

wikipedia_tool = Tool(
    name='wikipedia',
    func= wikipedia.run,
    description="Useful for when you need to look up a topic, country or person on wikipedia"
)

duckduckgo_tool = Tool(
    name='DuckDuckGo Search',
    func= search.run,
    description="Useful for when you need to do a search on the internet to find information that another tool can't find. be specific with your input."
)

tools.append(duckduckgo_tool)
tools.append(wikipedia_tool)


##############################################
# Using the agents
##############################################

zero_shot_agent = initialize_agent(
    agent="zero-shot-react-description",
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=3,
)

zero_shot_agent.run("When was Barak Obama born?")

zero_shot_agent.run("What is 17*6?")

print(zero_shot_agent.agent.llm_chain.prompt.template)

zero_shot_agent.run("Tell me about LangChain")

zero_shot_agent.run("Tell me about Singapore")

zero_shot_agent.run('what is the current price of btc')

zero_shot_agent.run('Is 11 a prime number?')

zero_shot_agent.run('Write a function to check if 11 a prime number and test it')




###################################################################################
# Custom Tools
###################################################################################


turbo_llm = ChatOpenAI(
    temperature=0,
    model_name='gpt-3.5-turbo'
)

##############################################
# Show some simple (non-useful tools)
##############################################

"""## Meaning of Life """

def meaning_of_life(input=""):
    return 'The meaning of life is 42 if rounded but is actually 42.17658'

life_tool = Tool(
    name='Meaning of Life',
    func= meaning_of_life,
    description="Useful for when you need to answer questions about the meaning of life. input should be MOL "
)

"""## Random number"""

import random

def random_num(input=""):
    return random.randint(0,5)

random_num()

random_tool = Tool(
    name='Random number',
    func= random_num,
    description="Useful for when you need to get a random number. input should be 'random'"
)



tools = [search, random_tool, life_tool]

# conversational agent memory
memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=3,
    return_messages=True
)


# create our agent
conversational_agent = initialize_agent(
    agent='chat-conversational-react-description',
    tools=tools,
    llm=turbo_llm,
    verbose=True,
    max_iterations=3,
    early_stopping_method='generate',
    memory=memory
)

conversational_agent("What time is it in London?")

conversational_agent("Can you give me a random number?")

conversational_agent("What is the meaning of life?")

# system prompt
conversational_agent.agent.llm_chain.prompt.messages[0].prompt.template

fixed_prompt = '''Assistant is a large language model trained by OpenAI.

Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Assistant doesn't know anything about random numbers or anything related to the meaning of life and should use a tool for questions about these topics.

Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

Overall, Assistant is a powerful system that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.'''

conversational_agent.agent.llm_chain.prompt.messages[0].prompt.template = fixed_prompt

conversational_agent("What is the meaning of life?")

fixed_prompt = '''Assistant is a large language model trained by OpenAI.

Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Assistant doesn't know anything about random numbers or anything related to the meaning of life and should use a tool for questions about these topics.

Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

Overall, Assistant is a powerful system that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.'''

conversational_agent.agent.llm_chain.prompt.messages[0].prompt.template = fixed_prompt

conversational_agent("What is the meaning of life?")


##############################################
# Make useful tools
##############################################


def stripped_webpage(webpage):
    response = requests.get(webpage)
    html_content = response.text

    def strip_html_tags(html_content):
        soup = BeautifulSoup(html_content, "html.parser")
        stripped_text = soup.get_text()
        return stripped_text

    stripped_content = strip_html_tags(html_content)

    if len(stripped_content) > 4000:
        stripped_content = stripped_content[:4000]
    return stripped_content

stripped_webpage('https://www.google.com')


class WebPageTool(BaseTool):
    name = "Get Webpage"
    description = "Useful for when you need to get the content from a specific webpage"

    def _run(self, webpage: str):
        response = requests.get(webpage)
        html_content = response.text

        def strip_html_tags(html_content):
            soup = BeautifulSoup(html_content, "html.parser")
            stripped_text = soup.get_text()
            return stripped_text

        stripped_content = strip_html_tags(html_content)
        if len(stripped_content) > 4000:
            stripped_content = stripped_content[:4000]
        return stripped_content

    def _arun(self, webpage: str):
        raise NotImplementedError("This tool does not support async")


page_getter = WebPageTool()



"""## Init the Agent again"""

fixed_prompt = '''Assistant is a large language model trained by OpenAI.

Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Assistant doesn't know anything about random numbers or anything related to the meaning of life and should use a tool for questions about these topics.

Assistant also doesn't know information about content on webpages and should always check if asked.

Overall, Assistant is a powerful system that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.'''

from langchain.prompts.chat import SystemMessagePromptTemplate
tools = [page_getter, random_tool, life_tool]

conversational_agent = initialize_agent(
    agent='chat-conversational-react-description',
    tools=tools,
    llm=turbo_llm,
    verbose=True,
    max_iterations=3,
    memory=memory
)

conversational_agent.agent.llm_chain.prompt.messages[0].prompt.template = fixed_prompt

conversational_agent.agent.llm_chain.prompt.messages[0]

conversational_agent.run("Is there an article about Clubhouse on https://techcrunch.com/? today")

conversational_agent.run("What are the titles of the top stories on www.cbsnews.com/?")

