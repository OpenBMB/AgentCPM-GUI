#!/usr/bin/env python3
# coding: utf-8
"""
Inference code is modified from the origin repo: https://github.com/xlang-ai/aguvis
See the origin code at:
Inference: https://github.com/xlang-ai/aguvis/blob/main/src/aguvis/serve/cli.py
Prompts: https://github.com/xlang-ai/aguvis/blob/main/src/aguvis/constants.py
"""

import os
import re
import sys
import json
import argparse
import random
import warnings
import multiprocessing
from tqdm import tqdm
from io import BytesIO
from typing import List, Literal, Optional, Dict
from concurrent.futures import ProcessPoolExecutor,as_completed,ThreadPoolExecutor


current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)

if current_dir not in sys.path:
    sys.path.append(current_dir)

# Ignore the warnings of eos_end_token.
warnings.filterwarnings("ignore") # Setting `pad_token_id` to `eos_token_id`:151658 for open-end generation. This warning.


import requests
from PIL import Image
from qwen_vl_utils import process_vision_info
from torch import NoneType
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from transformers import logging as tf_logging
tf_logging.set_verbosity_error()

import logging
logging.getLogger("transformers").setLevel(logging.WARNING)

from utils.utils import get_dataset_dir


DEVICES = [
    "cuda:0", "cuda:1", "cuda:2", "cuda:3",
    "cuda:4","cuda:5", "cuda:6", "cuda:7",
    ]

# Define prompt settings =============

CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

# System Message
grounding_system_message = "You are a GUI agent. You are given a task and a screenshot of the screen. You need to perform a series of pyautogui actions to complete the task."

# Chat Template
chat_template = "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"

assistant_template = "{% for message in messages %}{{'<|im_start|>' + message['role']}}{% if 'recipient' in message %}<|recipient|>{{ message['recipient'] }}{% endif %}{{'\n' + message['content'][0]['text']}}{% if 'end_turn' in message and message['end_turn'] %}{{'<|diff_marker|>\n'}}{% else %}{{'<|im_end|>\n'}}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant<|recipient|>' }}{% endif %}"

# Special Tokens
additional_special_tokens = [
    "<|im_start|>",
    "<|im_end|>",
    "<|object_ref_start|>",
    "<|object_ref_end|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|quad_start|>",
    "<|quad_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|vision_pad|>",
    "<|image_pad|>",
    "<|video_pad|>",
    "<|recipient|>",
    "<|diff_marker|>",
]

# Plugin Functions
select_option_func = {
    "name": "browser.select_option",
    "description": "Select an option from a dropdown menu",
    "parameters": {
        "type": "object",
        "properties": {
            "x": {
                "type": "number",
                "description": "The x coordinate of the dropdown menu",
            },
            "y": {
                "type": "number",
                "description": "The y coordinate of the dropdown menu",
            },
            "value": {
                "type": "string",
                "description": "The value of the option to select",
            },
        },
        "required": ["x", "y", "value"],
    },
}

swipe_func = {
    "name": "mobile.swipe",
    "description": "Swipe on the screen",
    "parameters": {
        "type": "object",
        "properties": {
            "from_coord": {
                "type": "array",
                "items": {"type": "number"},
                "description": "The starting coordinates of the swipe",
            },
            "to_coord": {
                "type": "array",
                "items": {"type": "number"},
                "description": "The ending coordinates of the swipe",
            },
        },
        "required": ["from_coord", "to_coord"],
    },
}

home_func = {"name": "mobile.home", "description": "Press the home button"}

back_func = {"name": "mobile.back", "description": "Press the back button"}

wait_func = {
    "name": "mobile.wait",
    "description": "wait for the change to happen",
    "parameters": {
        "type": "object",
        "properties": {
            "seconds": {
                "type": "number",
                "description": "The seconds to wait",
            },
        },
        "required": ["seconds"],
    },
}

long_press_func = {
    "name": "mobile.long_press",
    "description": "Long press on the screen",
    "parameters": {
        "type": "object",
        "properties": {
            "x": {
                "type": "number",
                "description": "The x coordinate of the long press",
            },
            "y": {
                "type": "number",
                "description": "The y coordinate of the long press",
            },
        },
        "required": ["x", "y"],
    },
}

open_app_func = {
    "name": "mobile.open_app",
    "description": "Open an app on the device",
    "parameters": {
        "type": "object",
        "properties": {
            "app_name": {
                "type": "string",
                "description": "The name of the app to open",
            },
        },
        "required": ["app_name"],
    },
}

agent_system_message = f"""You are a GUI agent. You are given a task and a screenshot of the screen. You need to perform a series of pyautogui actions to complete the task.

You have access to the following functions:
- {json.dumps(swipe_func)}
- {json.dumps(home_func)}
- {json.dumps(back_func)}
- {json.dumps(wait_func)}
- {json.dumps(long_press_func)}
- {json.dumps(open_app_func)}
"""

user_instruction = """Please generate the next move according to the ui screenshot, instruction and previous actions.

Instruction: {overall_goal}

Previous actions:
{previous_actions}
"""

until = ["<|diff_marker|>"]


# Prompt setting ends. =============

# action mapping begins. ===========
def mapping_actions(episode: dict) -> dict:
    """
    Mapping the string from aguvis model into minicpm Action space.

    Args:
        episode (dict): The episode dict, containing all information and a prediction string.

    Returns:
        the episode whose prediction string is mapped into minicpm Action space.

    In practice the model will output unstable strings. We only handle those stable cases.
    """

    FAIL_PARSE = {
        "STATUS": "FAIL"
    }

    action:str = episode["pred"].split('\n')[-1].strip()

    platform = action.split('.')[0]

    function = action[len(platform) + 1 :]

    if platform == "pyautogui":

        if function.startswith("click"):
            # deal with click function.
            try:
                matches = re.findall(r"[-+]?\d*\.\d+|\d+", function)
                x,y = matches
                x = round(float(x) * 1000)
                y = round(float(y) * 1000)

                episode["pred"] = {
                    "POINT": [x, y],
                    "duration": 200,
                    "STATUS": "continue"
                }
            except Exception as e:
                print(f"Failed to parse POINT ACTION {function}: {e}")
                episode["pred"] = FAIL_PARSE

        elif function.startswith("write"):
            # deal with type action.
            try:

                pattern = r'message=(["\'])(.*?)\1'
                match = re.search(pattern, function)

                text = match.group(2)

                episode["pred"] = {
                    "TYPE": text,
                    "duration": 200,
                    "STATUS": "continue"
                }

            except Exception as e:
                print(f"Failed to parse TYPE ACTION {function}: {e}")
                episode["pred"] = FAIL_PARSE

        elif function.startswith("scroll"):
            # deal with scroll up/down
            try:
                pattern = r'scroll\(page=([-+]?\d*\.\d+|\d+)\)'
                match = re.match(pattern, function)

                value = float(match.group(1))

                episode["pred"] = {
                    "POINT": [500, 500],
                    "to": "up" if value > 0 else "down",
                    "duration": 200,
                    "STATUS": "continue"
                }

            except Exception as e:
                print(f"Failed to parse MOVE_TO ACTION {function}: {e}")
                episode["pred"] = FAIL_PARSE

        elif function.startswith("hscroll"):
            # deal with scroll left/right
            try:
                pattern = r'hscroll\(page=([-+]?\d*\.\d+|\d+)\)'
                match = re.match(pattern, function)

                value = float(match.group(1))

                episode["pred"] = {
                    "POINT": [500, 500],
                    "to": "left" if value < 0 else "right",
                    "duration": 200,
                    "STATUS": "continue"
                }

            except Exception as e:
                print(f"Failed to parse MOVE_TO ACTION {function}: {e}")
                episode["pred"] = FAIL_PARSE

        else:
            print(f"Unrecognize action in {platform}: {function}")
            episode["pred"] = FAIL_PARSE

    elif platform == "mobile":

        if function.startswith("back"):
            # deal with back action.
            episode["pred"] = {
                "PRESS": "BACK",
                "duration": 200,
                "STATUS": "continue"
            }
        elif function.startswith("home"):
            # deal with home action.
            episode["pred"] = {
                "PRESS": "HOME",
                "duration": 200,
                "STATUS": "continue"
            }

        elif function.startswith("terminate"):
            # deal with terminate action.
            if 'success' in action:
                episode["pred"] = {
                    "STATUS": "finish"
                }
            else:
                episode["pred"] = {
                    "STATUS": "interrupt"
                }

        elif function.startswith("open_app"):
            # deal with open_app action. This action will not be accepted by our evaluation.
            try:
                match = re.search(r"app_name='([^']+)'", function)
                app_name = match.group(1)

                episode["pred"] = {
                    "open_app": app_name,
                    "duration": 200,
                    "STATUS": "continue"
                }

            except Exception as e:
                print(f"Failed to parse open_app ACTION {function}: {e}")
                episode["pred"] = FAIL_PARSE

        elif function.startswith("wait"):
            # deal with wait action.
            episode["pred"] = {
                "duration": 3000,
                "STATUS": "continue"
            }

        elif function.startswith("long_press"):
            # deal with long_press action.
            try:
                matches = re.findall(r"[-+]?\d*\.\d+|\d+", action)
                x,y = matches
                x = round(float(x) * 1000)
                y = round(float(y) * 1000)

                episode["pred"] = {
                    "POINT": [x, y],
                    "duration": 1000,
                    "STATUS": "continue"
                }

            except Exception as e:
                print(f"Failed to parse LONG_PRESS ACTION {function}: {e}")
                episode["pred"] = FAIL_PARSE

        else:
            print(f"Unrecognize action in {platform}: {function}")
            episode["pred"] = FAIL_PARSE

    else:
        # Any other unstable output will be informed.
        print(f'Unrecognize output: {repr(episode["pred"])}.')
        episode["pred"] = FAIL_PARSE

    return episode

# action mapping ends. =============

_llm: Optional[Qwen2VLForConditionalGeneration] = None
_processor: Optional[Qwen2VLProcessor] = None
_tokenizer = None


def _init_llm(model_name:str) -> None:
    """
    load the models and its relative processor, tokenizer.

    Args:
        model_name (str): the model name or model path.

    Returns:
        None
    """

    global _llm, _processor, _tokenizer

    if _llm is None:
        _llm = Qwen2VLForConditionalGeneration.from_pretrained(model_name, trust_remote_code=True)

    if _processor is None:
        _processor = Qwen2VLProcessor.from_pretrained(model_name, trust_remote_code=True)

    if _tokenizer is None:
        _tokenizer = _processor.tokenizer


def move_to(device):
    """
    Move the model to the specified device.

    Args:
        device (str): The device to move the model to.

    Returns:
        None
    """
    global _llm,_tokenizer
    if _llm is None:
        raise ValueError("Error, LLM is not initialized.")
    _llm = _llm.to(device)
    if _processor is None:
        raise ValueError("Error, Processor is not initialized.")
    if _tokenizer is None:
        raise ValueError("Error, Tokenizer is not initialized.")
    return f"Moved to {device}"


def process_data(episode:Dict,
                image_path: str,
                args:argparse.Namespace) -> Dict:
    """
    Process the data to run predict.

    Args:
        episode (Dict): one single data.
        image_path (str): the image_path processed.
        args(argparse.Namespace): the args passed in from terminal.

    Returns:
        Dict: the param used to run predict function.
    """

    def load_image(image_file: str) -> Image:
        """
        Origin code for loading the image.

        Args:
            image_file (str): the path or the url of the image.

            Note we didn't do any resize action here.

        Returns:
            Image: the Image class.
        """
        if image_file.startswith(("http://", "https://")):
            response = requests.get(image_file)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(image_file).convert("RGB")
        return image

    image: Image = load_image(image_path)
    instruction:str = episode['instruction']
    low_instruction: Optional[str] = None

    data_name:str = args.data_name

    if data_name == 'android_control_low_test':
        low_instruction:str = episode['low_instruction']

    return {
        "image": image,
        "episode": episode,
        "instruction": instruction,
        "previous_actions": None,
        "low_level_instruction": low_instruction,
        "mode": args.mode,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens
    }


def generate_response(
    image: Image.Image,
    episode: Dict,
    instruction: str,
    previous_actions: Optional[str | List[str]] = None,
    low_level_instruction: Optional[str] = None,
    mode: Literal["self-plan", "force-plan", "grounding"] = "self-plan",
    temperature: float = 0,
    max_new_tokens: int = 1024) -> str:
    """
    Modified from the origin code. Do the inference based on the params.

    Args:
        image (Image.Image)
        instruction (str)
        previous_actions (Optional[str  |  List[str]], optional)
        low_level_instruction (Optional[str], optional)
        mode (Literal[], optional)
        temperature (float, optional)
        max_new_tokens (int, optional)

    Returns:
        str: the predict result.
    """

    global _llm, _processor, _tokenizer

    # process the prompt.
    system_message = {
        "role": "system",
        "content": grounding_system_message if mode == "grounding" else agent_system_message,
    }

    if isinstance(previous_actions, list):
        previous_actions = "\n".join(previous_actions)
    if not previous_actions:
        previous_actions = "None"

    user_message = {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {
                "type": "text",
                "text": user_instruction.format(
                    overall_goal=instruction,
                    previous_actions=previous_actions,
                    low_level_instruction=low_level_instruction,
                ),
            },
        ],
    }

    if low_level_instruction:
        recipient_text = f"<|im_start|>assistant<|recipient|>all\nAction: {low_level_instruction}\n"
    elif mode == "grounding":
        recipient_text = "<|im_start|>assistant<|recipient|>os\n"
    elif mode == "self-plan":
        recipient_text = "<|im_start|>assistant<|recipient|>"
    elif mode == "force-plan":
        recipient_text = "<|im_start|>assistant<|recipient|>all\nThought: "
    else:
        raise ValueError(f"Invalid mode: {mode}")

    messages = [system_message, user_message]
    text = _processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, chat_template=chat_template
    )
    text += recipient_text
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = _processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(_llm.device)
    cont = _llm.generate(**inputs, temperature=temperature, max_new_tokens=max_new_tokens)

    cont_toks = cont.tolist()[0][len(inputs.input_ids[0]) :]
    text_outputs = _tokenizer.decode(cont_toks, skip_special_tokens=True).strip()
    for term in until:
        if term:
            text_outputs = text_outputs.split(term)[0]

    episode["pred"] = text_outputs

    return episode


def predict(args:argparse.Namespace):
    """
    Entry function, predict based on the given args.

    Args:
        args
    """
    args.data_dir, args.split, data_subset = get_dataset_dir(args.data_name)
    print(f"Predicting on: {args.data_dir}/{args.split}")
    print(f"Data subset: {data_subset}")

    if multiprocessing.get_start_method(allow_none=True) != "spawn":
        multiprocessing.set_start_method("spawn", force=True)

    with ProcessPoolExecutor(max_workers=len(DEVICES),initializer=_init_llm,initargs=(args.model_path,)) as poolexec:
        tasks = []
        print("Moving model to devices")
        futures = [poolexec.submit(move_to, dev) for dev in DEVICES]
        for fut in futures: print(fut.result())

        for dataset in data_subset:
            save_dir = os.path.join(args.output_dir, dataset)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            episode_dir = os.path.join(args.data_dir, args.split, dataset)
            output_file = os.path.join(save_dir, "predict.jsonl")

            # Get the list of all episodes files
            if os.path.exists(episode_dir):
                episodes_files = os.listdir(episode_dir)
            else:
                continue

            future = []
            all_tasks = []

            print("Loading episodes")
            with ThreadPoolExecutor(max_workers=16) as executor:
                for episodes_file in episodes_files:

                    episodes_path = os.path.join(episode_dir, episodes_file, f"{episodes_file}.json")
                    try:
                        with open(episodes_path, 'r', encoding='utf-8') as f:
                            episodes = json.load(f)
                    except Exception as e:
                        print(f"Failed to load {episodes_path}: {e}")
                        continue
                        # Skip this file on error

                    for episode in episodes:
                        episode["category"] = dataset
                        image_path = os.path.join(episode_dir, episodes_file, f"{episodes_file}_{episode['step_id']}.jpeg")
                        if not os.path.exists(image_path):
                            image_path = image_path.replace(".jpeg", ".png")
                            if not os.path.exists(image_path):
                                image_path = episode['image_path']
                        future.append(executor.submit(process_data, episode, image_path, args))

                for f in as_completed(future):
                    all_tasks.append(f.result())

            with open(output_file, "w", encoding="utf-8") as f_out:
                print("Predicting")
                tasks = []
                for task_value in all_tasks:
                    tasks.append(poolexec.submit(generate_response, **task_value))

                for task in tqdm(as_completed(tasks), total=len(tasks), dynamic_ncols=True):
                    try:
                        episode = task.result()
                        episode = mapping_actions(episode)
                        episode_json = json.dumps(episode, ensure_ascii=False)
                        f_out.write(episode_json + "\n")
                        f_out.flush()
                    except Exception as e:
                        print(f"Error: {e}")
                        continue
            print(f"Prediction saved at: {output_file}.")
    os.system(f"cat {args.output_dir}/*/predict.jsonl > {args.output_dir}/all.jsonl")
    print(f"Merged prediction saved at: {args.output_dir}/all.jsonl.")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2020, help="Random seed")
    parser.add_argument("--model_path", type=str, required=True, help="Model path")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--data_name", type=str, required=True, choices=['gui_odyssey_test', 'chinese_app_test', 'aitz_test', 'android_control_high_test', 'android_control_low_test'], help="Eval dataset name")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mode", type=str, choices = ["self-plan", "force-plan", "grounding"], default="self-plan")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=1024)

    args = parser.parse_args()
    random.seed(args.seed)

    print(f'Loading model at : {args.model_path}')
    print(f'Saving results at: {args.output_dir}')

    predict(args)