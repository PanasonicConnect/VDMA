import os
import json
from datetime import timedelta
from langchain.agents import tool


@tool
def dummy_tool() -> str:
    """
    This is dummy tool.

    Returns:
    str: 'hello world'
    """
    print ("called the dummy tool.")
    return "hello world"


@tool
def analyze_video(gpt_prompt:str) -> str:
    """
    Analyze video tool.

    Parameters:
    prompt (str): In the GPT prompt, You must include the Q&A you want to solve and the perspective from which you want GPT to view the video.For example, if you want GPT to watch the video from the perspective of a household expert, you could write: "You are a household expert. From the perspective of someone doing household chores, please consider which of the following option seems most plausible. Question...

    Returns:
    str: The analysis result.
    """

    from util import ask_gpt4_vision

    azure_openai_endpoint   = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_openai_api_key    = os.getenv("AZURE_OPENAI_API_KEY")
    azure_openai_version    = os.getenv("AZURE_OPENAI_VERSION")
    azure_openai_model_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    acv_base_url            = os.getenv("ACV_BASE_URL")
    acv_api_key             = os.getenv("ACV_API_KEY")
    video_index             = os.getenv("VIDEO_INDEX")
    video_sas_token         = os.getenv("VIDEO_SAS_TOKEN")

    print ("Called the tool of analyze_video.")
    print (gpt_prompt)
    return ask_gpt4_vision(
                openai_api_base_url=azure_openai_endpoint,
                openai_deployment_name=azure_openai_model_name,
                openai_api_key=azure_openai_api_key,
                openai_api_version=azure_openai_version,
                acv_base_url=acv_base_url,
                acv_api_key=acv_api_key,
                index_name=video_index,
                sas_url=video_sas_token,
                prompt_text=gpt_prompt
            )


@tool
def analyze_video_gpt4o(gpt_prompt:str) -> str:
    """
    Analyze video tool.

    Parameters:
    gpt_prompt (str): In the GPT prompt, You must include 5 questions based on original questions and options.
    For example, if the question asks about the purpose of the video and OptionA is “C is looking for a T-shirt” and OptionB is “C is cleaning up the room,
    OptionA is “C is looking for a T-shirt?” and OptionB is “C is tidying the room?” and so on. 
    The questions should be Yes/No questions whenever possible.
    Also, please indicate what role you would like the respondent to play in answering the questions.

    Returns:
    str: The analysis result.
    """

    from util import ask_gpt4_omni

    print ("gpt_prompt: ", gpt_prompt)

    openai_api_key          = os.getenv("OPENAI_API_KEY")
    video_file_name         = os.getenv("VIDEO_FILE_NAME")

    print ("Called the tool of analyze_video_gpt4o.")

    result = ask_gpt4_omni(
                openai_api_key=openai_api_key,
                prompt_text=gpt_prompt,
                image_dir="/home/project_ws/images",
                vid=video_file_name,
                temperature=0.7,
                frame_num=90
            )
    print ("result: ", result)
    return result


@tool
def retrieve_video_clip_captions(gpt_prompt:str) -> str:
    """
    Analyze captioning tool.

    Parameters:
    gpt_prompt (str): In the GPT prompt, You must include 5 questions based on original questions and options.
    For example, if the question asks about the purpose of the video and OptionA is “C is looking for a T-shirt” and OptionB is “C is cleaning up the room,
    OptionA is “C is looking for a T-shirt?” and OptionB is “C is tidying the room?” and so on. 
    The questions should be Yes/No questions whenever possible.
    Also, please indicate what role you would like the respondent to play in answering the questions.

    Returns:
    str: The analysis result.
    """

    print("Called the Image captioning tool.")

    video_filename = os.getenv("VIDEO_FILE_NAME")

    with open("/home/project_ws/EgoSchemaVQA/LLoVi/data/egoschema/lavila_fullset.json", "r") as f:
        captions_data = json.load(f)

    captions = captions_data.get(video_filename, [])
    result = []
    previous_caption = None

    for i, caption in enumerate(captions):

        # Remove the 'C' marker from the caption
        caption = caption.replace("#C ", "")
        caption = caption.replace("#c ", "")

        # Calculate the timestamp in hh:mm:ss format
        timestamp = str(timedelta(seconds=i))

        # Add the timestamp at the beginning of each caption
        timestamped_caption = f"{timestamp}: {caption}"

        # Add the caption to the result list if it's not a duplicate of the previous one
        if caption != previous_caption:
            result.append(timestamped_caption)

        # Update the previous caption
        previous_caption = caption

    prompt = "[Image Captions]\n"
    for caption in result:
        prompt += caption + "\n"

    prompt += "\n[Instructions]\n"
    prompt += gpt_prompt    

    print ("gpt_prompt: ", prompt)

    azure_openai_api_key    = os.getenv("AZURE_OPENAI_API_KEY")
    azure_openai_endpoint   = os.getenv("AZURE_OPENAI_ENDPOINT")

    from util import ask_gpt4
    result = ask_gpt4(
                    openai_deployment_name="gpt-4",
                    openai_api_version='2023-12-01-preview',
                    openai_api_key=azure_openai_api_key,
                    openai_api_base_url=azure_openai_endpoint,
                    prompt_text=prompt
                )
    print ("result: ", result)

    return result


# previous version
@tool
def retrieve_video_clip_captions_without_llm() -> list[str]:
    """
    Image captioning tool.

    Retrieve the captions of the specified video clip. Each caption is generated for notable changes within the video, helping in recognizing fine-grained changes and flow within the video. The captions include markers 'C' representing the person wearing the camera.

    Returns:
    list[str]: A list of captions for the video.
    """

    print("Called the Image captioning tool.")

    video_filename = os.getenv("VIDEO_FILE_NAME")

    with open("/home/project_ws/EgoSchemaVQA/LLoVi/data/egoschema/lavila_fullset.json", "r") as f:
        captions_data = json.load(f)

    captions = captions_data.get(video_filename, [])
    result = []
    previous_caption = None

    for i, caption in enumerate(captions):

        # Remove the 'C' marker from the caption
        caption = caption.replace("#C ", "")
        caption = caption.replace("#c ", "")

        # Calculate the timestamp in hh:mm:ss format
        timestamp = str(timedelta(seconds=i))

        # Add the timestamp at the beginning of each caption
        timestamped_caption = f"{timestamp}: {caption}"

        # Add the caption to the result list if it's not a duplicate of the previous one
        if caption != previous_caption:
            result.append(timestamped_caption)

        # Update the previous caption
        previous_caption = caption

    return result


if __name__ == "__main__":

    data = retrieve_video_clip_captions()
    for caption in data:
        print (caption)
    print ("length of data: ", len(data))