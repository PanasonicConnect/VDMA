import os
import json
import copy
import time
import random
from util import generate_sas_url
from util import select_data_and_mark_as_processing
from util import unmark_as_processing
from util import save_result
from stage1 import execute_stage1
from stage2 import execute_stage2


# Sleep for a random duration between 0 and 10 seconds
sleep_time = random.uniform(0, 10)
time.sleep(sleep_time)


QUESTION_FILE_PATH = "subset_anno.json" # Set the file path containing the question

azure_openai_endpoint   = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_openai_api_key    = os.getenv("AZURE_OPENAI_API_KEY")
azure_openai_version    = os.getenv("AZURE_OPENAI_VERSION")
azure_openai_model_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
acv_base_url            = os.getenv("ACV_BASE_URL")
acv_api_key             = os.getenv("ACV_API_KEY")
blob_account_name       = os.getenv("BLOB_ACCOUNT_NAME")
blob_account_key        = os.getenv("BLOB_ACCOUNT_KEY")
blob_container_name     = os.getenv("BLOB_CONTAINER_NAME")




def set_environment_variables(video_id:str, json_data:dict, use_re_writed_qa=False):
    index_name = "video-" + video_id[:8]
    sas_url    = ""#generate_sas_url(account_name=blob_account_name, account_key=blob_account_key, container_name=blob_container_name, blob_name=video_id)
    os.environ["VIDEO_INDEX"]     = index_name
    os.environ["VIDEO_SAS_TOKEN"] = sas_url
    os.environ["VIDEO_FILE_NAME"] = video_id

    if use_re_writed_qa == False:
        os.environ["QA_JSON_STR"] = json.dumps(json_data)
    else:
        json_data["rewrited_qa"]["truth"] = json_data["truth"]
        os.environ["QA_JSON_STR"] = json.dumps(json_data["rewrited_qa"])

    print ("{} : {}".format(video_id, index_name))
    print (sas_url)
    print ("use Re-writed QA") if use_re_writed_qa else print ("use Original QA")


# Loop through questions
while True:

    try:
        video_id, json_data = select_data_and_mark_as_processing(QUESTION_FILE_PATH)

        if video_id is None: # All data has been processed
            break

        # Set environment variables
        print ("****************************************")
        set_environment_variables(video_id, json_data, use_re_writed_qa=False)

        # Execute stage1
        print ("execute stage1")
        expert_info = execute_stage1()

        # Execute stage2
        print ("execute stage2")
        result, agent_response, agent_prompts = execute_stage2(expert_info)

        # Save result
        save_result(QUESTION_FILE_PATH, video_id, expert_info, agent_prompts, agent_response, result)

    except Exception as e:
        print ("Error: ", e)
        #unmark_as_processing(QUESTION_FILE_PATH, video_id)
        time.sleep(1)
        continue
