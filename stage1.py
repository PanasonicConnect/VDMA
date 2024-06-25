import os
import json
import time
from util import ask_gpt4
from util import ask_gpt4_vision
from util import ask_gpt4_omni
from util import create_mas_stage1_prompt
from util import extract_expert_info


def execute_stage1():

    azure_openai_endpoint   = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_openai_api_key    = os.getenv("AZURE_OPENAI_API_KEY")
    azure_openai_version    = os.getenv("AZURE_OPENAI_VERSION")
    azure_openai_model_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    openai_api_key          = os.getenv("OPENAI_API_KEY")
    acv_base_url            = os.getenv("ACV_BASE_URL")
    acv_api_key             = os.getenv("ACV_API_KEY")
    video_index             = os.getenv("VIDEO_INDEX")
    video_sas_token         = os.getenv("VIDEO_SAS_TOKEN")
    video_filename          = os.getenv("VIDEO_FILE_NAME")
    qa_json_str             = os.getenv("QA_JSON_STR")

    question = json.loads(qa_json_str)

    prompt = create_mas_stage1_prompt(question)
    print (prompt)

    ## azure gpt4-vision-preview
    # response_data = ask_gpt4_vision(
    #             openai_api_base_url=azure_openai_endpoint,
    #             openai_deployment_name=azure_openai_model_name,
    #             openai_api_key=azure_openai_api_key,
    #             openai_api_version=azure_openai_version,
    #             acv_base_url=acv_base_url,
    #             acv_api_key=acv_api_key,
    #             index_name=video_index,
    #             sas_url=video_sas_token,
    #             prompt_text=prompt
    #         )

    ## azure gpt4-turbo
    # response_data = ask_gpt4(
    #             openai_deployment_name="gpt-4",
    #             openai_api_version='2023-12-01-preview',
    #             openai_api_key=azure_openai_api_key,
    #             openai_api_base_url=azure_openai_endpoint,
    #             prompt_text=prompt
    #         )

    response_data = ask_gpt4_omni(
                openai_api_key=openai_api_key,
                prompt_text=prompt,
                image_dir="/home/project_ws/images",
                vid=video_filename,
                temperature=0.7
            )

    expert_info = extract_expert_info(response_data)
    if not expert_info:
        print ("**** Expert info is empty. Re-running the stage1. ****")
        time.sleep(3) # sleep for 3 second to avoid the rate limit
        return execute_stage1()

    print ("*********** Stage1 Result **************")
    print(json.dumps(expert_info, indent=2, ensure_ascii=False))
    print ("****************************************")

    return expert_info


if __name__ == "__main__":

    execute_stage1()