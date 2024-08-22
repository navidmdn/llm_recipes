from fire import Fire
import numpy as np
from fire import Fire
import os
import json
from typing import List, Dict
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import StrOutputParser
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import glob
from tqdm import tqdm
from langchain_chroma import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
# from utils import load_hf_embedding_model
from langchain_openai import ChatOpenAI

def simulate(persona1: str, persona2: str, scenario: str, llm: Runnable, n_turns: int) -> None:
    """

    Simulate a conversation between two speakers based on their personas and a scenario.

    :param persona1: the first speaker's persona
    :param persona2: the second speaker's persona
    :param scenario: the scenario
    :return:
    """

    person1_history = []
    person2_history = []

    person1_sys_msg = SystemMessage(content=f"Your persona: {persona1}\n\nScenario: {scenario}")
    person2_sys_msg = SystemMessage(content=f"Your persona: {persona2}\n\nScenario: {scenario}")

    cur_speaker = np.random.choice([1, 2])

    last_question = "What are your key arguments for this approach, and how do you envision it leading to a better future for Iran?"
    print(f"speaker {3-cur_speaker}: {last_question}")

    for _ in range(n_turns):
        if cur_speaker == 1:
            prompt_tmp = ChatPromptTemplate.from_messages([
                person1_sys_msg,
                *person1_history,
                HumanMessagePromptTemplate.from_template("{last_utt}")
            ])

            person1_history.append(HumanMessage(content=last_question))
            person2_history.append(AIMessage(content=last_question))
            cur_speaker = 2
        else:
            prompt_tmp = ChatPromptTemplate.from_messages([
                person2_sys_msg,
                *person2_history,
                HumanMessagePromptTemplate.from_template("{last_utt}")
            ])
            person2_history.append(HumanMessage(content=last_question))
            person1_history.append(AIMessage(content=last_question))

            cur_speaker = 1

        # print("Prompt: ", prompt_tmp.invoke({"last_utt": last_question}))
        chain = prompt_tmp | llm | StrOutputParser()

        response = chain.invoke({"last_utt": last_question})
        print(f"speaker {3-cur_speaker}: {response}")
        last_question = response




def run(persona1_file: str = 'p1-test.txt', persona2_file: str = 'p2-test.txt', scenario_file: str = 'scenario-test.txt',
        n_iters: int = 1, llm_name: str = 'gpt-4o') -> None:

    """

    Simulate some conversations between two speakers based on their personas and a scenario and collects the conversation.

    :param llm_name:
    :param n_iters: number of iterations
    :param persona1_file: path to the first speaker's persona file
    :param persona2_file: path to the second speaker's persona file
    :param scenario_file: path to the scenario file
    :return:
    """

    with open(persona1_file, 'r') as f:
        persona1 = f.read().strip()

    with open(persona2_file, 'r') as f:
        persona2 = f.read().strip()

    with open(scenario_file, 'r') as f:
        scenario = f.read().strip()

    if 'llama' in llm_name or 'mistral' in llm_name:
        pass
    elif 'gpt' in llm_name:
        llm = ChatOpenAI(model_name=llm_name, openai_api_key=OPENAI_API_KEY, openai_api_base=OPENAI_API_BASE,
                        max_tokens=4096, temperature=0.7,)
    else:
        raise ValueError("LLM not implemented")

    for i in range(n_iters):
        simulate(persona1, persona2, scenario, llm, n_turns=10)


if __name__ == '__main__':
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    OPENAI_API_BASE = os.environ.get('OPENAI_API_BASE')

    Fire(run)