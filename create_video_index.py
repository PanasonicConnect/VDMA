import os
import requests
import json
import copy
import time
from util import *


# ***************** Configuration *****************
# QUESTION_FILE_PATH = "subset_anno.json"
QUESTION_FILE_PATH = "fullset_anno.json"

# Azure ComputerVision and BlobStorage Configuration
VISION_API_ENDPOINT = "" # ex. https://YOUR_RESOURCE_NAME.cognitiveservices.azure.com/
VISION_API_KEY      = "" # ex. 1234567890abcdef1234567890abcdef
BLOB_ACCOUNT_NAME   = "" # ex. YOUR_STORAGE_ACCOUNT_NAME
BLOB_ACCOUNT_KEY    = "" # ex. YOUR_STORAGE_ACCOUNT_KEY
BLOB_CONTAINER_NAME = "" # ex. YOUR_CONTAINER_NAME


# Delete all existing Azure Computer Vision indexes
# delete_all_video_index(VISION_API_ENDPOINT, VISION_API_KEY)
# print("delete_all_video_index done.")
# print("wait 30 seconds.")
# time.sleep(30)


# Load questions
with open(QUESTION_FILE_PATH, "r") as f:
    questions = json.load(f)


# Loop through questions
for i, (video_id, json_data) in enumerate(questions.items()):

    print ("------------------------------------")

    # Set index name
    index_name = "video-" + video_id[:8]
    print ("{} : {}".format(i, index_name))

    # Step 1 : Get SAS URL
    sas_url = generate_sas_url(account_name=BLOB_ACCOUNT_NAME, account_key=BLOB_ACCOUNT_KEY, container_name=BLOB_CONTAINER_NAME, blob_name=video_id)
    if sas_url is None:
        print("Failed to generate SAS URL.")
        continue

    if check_index_exists(VISION_API_ENDPOINT, VISION_API_KEY, index_name) == False:

        # Step 2 : Create an Index
        response = create_video_index(VISION_API_ENDPOINT, VISION_API_KEY, index_name)
        print(response.status_code, response.text)
        print ("create_video_index done.")

        # Step 3 : Add a video file to the index
        response = add_video_to_index(VISION_API_ENDPOINT, VISION_API_KEY, index_name, sas_url)
        print(response.status_code, response.text)
        print ("add_video_to_index done.")

        # Step 4 : Wait for ingestion to complete
        if not wait_for_ingestion_completion(VISION_API_ENDPOINT, VISION_API_KEY, index_name):
            print("Ingestion did not complete within the expected time.")
        print ("wait_for_ingestion_completion done.")
