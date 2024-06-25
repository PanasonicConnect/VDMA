import os
import sys
import time
import json
import operator
import functools

from langgraph.graph import StateGraph, END
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import AzureChatOpenAI, OpenAI, ChatOpenAI

from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict
from tools import analyze_video, retrieve_video_clip_captions, analyze_video_gpt4o, dummy_tool
from util import post_process, ask_gpt4, create_stage2_agent_prompt, create_stage2_organizer_prompt, create_question_sentence


azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_openai_api_key  = os.getenv("AZURE_OPENAI_API_KEY")
openai_api_key        = os.getenv("OPENAI_API_KEY")

tools = [analyze_video_gpt4o, retrieve_video_clip_captions]

# llm   = AzureChatOpenAI(
#     azure_deployment='gpt-4',
#     api_version='2023-12-01-preview',
#     azure_endpoint=azure_openai_endpoint,
#     api_key=azure_openai_api_key,
#     temperature=0.7,
#     streaming=False
#     )

llm = ChatOpenAI(
    api_key=openai_api_key,
    model='gpt-4o',
    temperature=0.0,
    streaming=False
    )


def create_agent(llm, tools: list, system_prompt: str):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
            ),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor


def agent_node(state, agent, name):
    print ("****************************************")
    print(f" Executing {name} node!")
    print ("****************************************")
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str


def mas_result_to_dict(result_data):
    log_dict = {}
    for message in result_data["messages"]:
        log_dict[message.name] = message.content
    return log_dict


def execute_stage2(expert_info):

    members = ["agent1", "agent2", "agent3", "organizer"]
    system_prompt = (
        "You are a supervisor who has been tasked with answering a quiz regarding the video. Work with the following members {members} and provide the most promising answer.\n"
        "Respond with FINISH along with your final answer. Each agent has one opportunity to speak, and the organizer should make the final decision."
        )

    options = ["FINISH"] + members
    function_def = {
        "name": "route",
        "description": "Select the next role.",
        "parameters": {
            "title": "routeSchema",
            "type": "object",
            "properties": {"next": {"title": "Next", "anyOf": [{"enum": options}]}},
            "required": ["next"],
        },
    }
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
                "Given the conversation above, who should act next?"
                " Or should we FINISH? Select one of: {options}"
                " If you want to finish the conversation, type 'FINISH' and Final Answer."
                ,
            ),
        ]
    ).partial(options=str(options), members=", ".join(members))

    supervisor_chain = (
        prompt
        | llm.bind_functions(functions=[function_def], function_call="route")
        | JsonOutputFunctionsParser()
    )

    # Load taget question
    qa_json_str = os.getenv("QA_JSON_STR")
    video_filename  = os.getenv("VIDEO_FILE_NAME")
    target_question_data = json.loads(qa_json_str)

    print ("****************************************")
    print (" Next Question: {}".format(video_filename))
    print ("****************************************")
    print (create_question_sentence(target_question_data))

    agent1_prompt = create_stage2_agent_prompt(target_question_data, expert_info["ExpertName1Prompt"], shuffle_questions=False)
    agent1 = create_agent(llm, tools, system_prompt=agent1_prompt)
    agent1_node = functools.partial(agent_node, agent=agent1, name="agent1")

    agent2_prompt = create_stage2_agent_prompt(target_question_data, expert_info["ExpertName2Prompt"], shuffle_questions=False)
    agent2 = create_agent(llm, tools, system_prompt=agent2_prompt)
    agent2_node = functools.partial(agent_node, agent=agent2, name="agent2")

    agent3_prompt = create_stage2_agent_prompt(target_question_data, expert_info["ExpertName3Prompt"], shuffle_questions=False)
    agent3 = create_agent(llm, [retrieve_video_clip_captions], system_prompt=agent3_prompt)
    agent3_node = functools.partial(agent_node, agent=agent3, name="agent3")

    organizer_prompt = create_stage2_organizer_prompt(target_question_data, shuffle_questions=False)
    organizer_agent = create_agent(llm, [dummy_tool], system_prompt=organizer_prompt)
    organizer_node = functools.partial(agent_node, agent=organizer_agent, name="organizer")

    # for debugging
    agent_prompts = {
        "agent1_prompt": agent1_prompt,
        "agent2_prompt": agent2_prompt,
        "agent3_prompt": agent3_prompt,
        "organizer_prompt": organizer_prompt
    }

    print ("******************** Agent1 Prompt ********************")
    print (agent1_prompt)
    print ("******************** Agent2 Prompt ********************")
    print (agent2_prompt)
    print ("******************** Agent3 Prompt ********************")
    print (agent3_prompt)
    print ("******************** Organizer Prompt ********************")
    print (organizer_prompt)
    print ("****************************************")
    # return

    # Create the workflow
    workflow = StateGraph(AgentState)
    workflow.add_node("agent1", agent1_node)
    workflow.add_node("agent2", agent2_node)
    workflow.add_node("agent3", agent3_node)
    workflow.add_node("organizer", organizer_node)
    workflow.add_node("supervisor", supervisor_chain)

    # Add edges to the workflow
    for member in members:
        workflow.add_edge(member, "supervisor")
    conditional_map = {k: k for k in members}
    conditional_map["FINISH"] = END
    workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)
    workflow.set_entry_point("supervisor")
    graph = workflow.compile()

    # Execute the graph
    # input_message = create_question_sentence(target_question_data) + "\n\nExclude options that contain unnecessary embellishments, such as subjective adverbs or clauses that cannot be objectively determined, and consider only the remaining options."
    input_message = create_question_sentence(target_question_data)
    print ("******** Stage2 input_message **********")
    print (input_message)
    print ("****************************************")
    agents_result = graph.invoke(
        {"messages": [HumanMessage(content=input_message, name="system")], "next": "agent1"},
        {"recursion_limit": 20}
    )

    prediction_num = post_process(agents_result["messages"][-1].content)
    if prediction_num == -1:
        prompt = agents_result["messages"][-1].content + "\n\nPlease retrieve the final answer from the sentence above. Your response should be one of the following options: Option A, Option B, Option C, Option D, Option E."
        response_data = ask_gpt4(openai_deployment_name="gpt-4", openai_api_version='2023-12-01-preview', openai_api_key=azure_openai_api_key, openai_api_base_url=azure_openai_endpoint, prompt_text=prompt)
        prediction_num = post_process(response_data)
    if prediction_num == -1:
        print ("***********************************************************")
        print ("Error: The result is -1. So, retry the stage2.")
        print ("***********************************************************")
        time.sleep(1)
        return execute_stage2(expert_info)

    agents_result_dict = mas_result_to_dict(agents_result)

    print ("*********** Stage2 Result **************")
    print(json.dumps(agents_result_dict, indent=2, ensure_ascii=False))
    print ("****************************************")
    print(f"Truth: {target_question_data['truth']}, Pred: {prediction_num} (Option{['A', 'B', 'C', 'D', 'E'][prediction_num]})" if 0 <= prediction_num <= 4 else "Error: Invalid result_data value")
    print ("****************************************")

    return prediction_num, agents_result_dict, agent_prompts


if __name__ == "__main__":

    data = {
        "ExpertName1": "Culinary Expert",
        "ExpertName1Prompt": "You are a Culinary Expert. Watch the video from the perspective of a professional chef and answer the following questions based on your expertise. Please think step-by-step.",
        "ExpertName2": "Kitchen Equipment Specialist",
        "ExpertName2Prompt": "You are a Kitchen Equipment Specialist. Watch the video from the perspective of an expert in kitchen tools and equipment and answer the following questions based on your expertise. Please think step-by-step.",
        "ExpertName3": "Home Cooking Enthusiast",
        "ExpertName3Prompt": "You are a Home Cooking Enthusiast. Watch the video from the perspective of someone who loves cooking at home and answer the following questions based on your expertise. Please think step-by-step."
    }

    execute_stage2(data)