import os
import json
from PIL import Image
from decord import VideoReader, cpu


def convert_videos_to_images(json_path, video_dir, save_dir):
    with open(json_path, "r") as f:
        questions = json.load(f)

    for i, data in enumerate(questions):
        print(i/len(questions) * 100, "%" )
        key = data["q_uid"]
        video_path = os.path.join(video_dir, key + ".mp4")
        frames = VideoReader(video_path, ctx=cpu(0))
        os.makedirs(os.path.join(save_dir, key), exist_ok=True)
        for i in range(int(len(frames) / 30)):
            img = Image.fromarray(frames[30 * i].asnumpy())
            img.save(os.path.join(save_dir, key, "{}_{:04}.jpg".format(key, i + 1)))

convert_videos_to_images("questions.json", "/mnt/ms1_nas/public/Ego4D/egoschema/videos", "/mnt/ms1_nas/public/Ego4D/egoschema/images")
