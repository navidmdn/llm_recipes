from fire import Fire
import numpy as np
from fire import Fire
import os
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from utils.langchain import load_hf_auto_regressive_model
# from langchain_openai import ChatOpenAI

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

    person1_sys_msg = SystemMessage(content=f"{persona1}")
    person2_sys_msg = SystemMessage(content=f"{persona2}")

    cur_speaker = np.random.choice([1, 2])

    last_question = "Hey! how's it going?"
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
        n_iters: int = 1, llm_name: str = 'gpt-4o', temperature=0.1, max_new_tokens=4096, hf=False, load_in_4bit=False,
        cache_dir=None) -> None:


    with open(persona1_file, 'r') as f:
        persona1 = f.read().strip()

    with open(persona2_file, 'r') as f:
        persona2 = f.read().strip()

    # with open(scenario_file, 'r') as f:
    #     scenario = f.read().strip()
    scenario = ""

    if 'llama' in llm_name or 'mistral' in llm_name:
        if hf:
            llm = load_hf_auto_regressive_model(llm_name, max_new_tokens=max_new_tokens, load_in_4bit=load_in_4bit,
                                                cache_dir=cache_dir)
        else:
            raise ValueError("LLM not implemented")
    elif 'gpt' in llm_name:
        llm = ChatOpenAI(model_name=llm_name, openai_api_key=OPENAI_API_KEY, openai_api_base=OPENAI_API_BASE,
                        max_tokens=max_new_tokens, temperature=temperature,)
    else:
        raise ValueError("LLM not implemented")

    for i in range(n_iters):
        simulate(persona1, persona2, scenario, llm, n_turns=10)


if __name__ == '__main__':
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    OPENAI_API_BASE = os.environ.get('OPENAI_API_BASE')

    Fire(run)