import base64
import sys
import argparse
import json
import re
import copy
import multiprocessing
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_ALLOW_DEPRECATED_BEAM_SEARCH"]="1"
import yaml
import time
import torch
import random
from yacs.config import CfgNode as CN
import re
import numpy as np
import requests
import jsonschema
from tqdm import tqdm
from transformers import AutoTokenizer,AutoModelForCausalLM
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from utils.qwen_mobile_tool import aitw_2_uitars
from concurrent.futures import ProcessPoolExecutor,as_completed,ThreadPoolExecutor
from PIL import Image
from utils.utils import get_dataset_dir
DEVICES = [
    "cuda:0", 
    "cuda:1", 
    #"cuda:2", "cuda:3",
    #   "cuda:4", "cuda:5", "cuda:6", "cuda:7",
    ]
torch.set_num_threads(4)
USE_LOW_INSTRUCTION = False

# Get the absolute path of the current file
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)

# Add the current file's directory to sys.path
if current_dir not in sys.path:
    sys.path.append(current_dir)
    

def compact_json_dumps(obj):
    return json.dumps(obj, indent=None, separators=(",", ":"), ensure_ascii=False)


NO_THOUGHT_EXAMPLE = {"Press":"BACK"}
SYSTEM_PROMPT = "You are a helpful assistant."


_llm = None
_tokenizer = None

def _init_llm(model_name):
    global _llm,_tokenizer
    if _llm is None:
        _llm = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,   trust_remote_code=True,  torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
    if _tokenizer is None:
        _tokenizer = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

def move_to(device):
    global _llm,_tokenizer
    if _llm is None:
        raise ValueError("Error, LLM is not initialized.")
    _llm = _llm.to(device)
    if _tokenizer is None:
        raise ValueError("Error, Tokenizer is not initialized.")
    # return _llm,_tokenizer
    return f"Moved to {device}"
def build_history_actions_str(history_list):
    history = ""
    for i, step_history in enumerate(history_list):
        history += f"Step {i+1}: {step_history['low_instruction']}\n"
    return history
def run_episode_high(episode, image_path,history_list,use_low_instruction):
    #print(msg)
    #print(episode)
    try:
        global _llm,_tokenizer
        torch.cuda.empty_cache()
        # msg[0]["content"].append(img)
        instruction = episode["instruction"]
        #low_instruction = episode["low_instruction"]
        #thought = "Thought: "+low_instruction+"\nAction:"
        #history = build_history_actions_str(history_list)
        history = ""
        a11y_tree = ""
        text = f"""\nYou are a foundational action model capable of automating tasks across various digital environments, including desktop systems like Windows, macOS, and Linux, as well as mobile platforms such as Android and iOS. You also excel in web browser environments. You will interact with digital devices in a human-like manner: by reading screenshots, analyzing them, and taking appropriate actions.\n\nYour expertise covers two types of digital tasks:\n    - Grounding: Given a screenshot and a description, you assist users in locating elements mentioned. Sometimes, you must infer which elements best fit the description when they aren't explicitly stated.\n    - Executable Language Grounding: With a screenshot and task instruction, your goal is to determine the executable actions needed to complete the task.\n\n\nYou are now operating in Executable Language Grounding mode. Your goal is to help users accomplish tasks by suggesting executable actions that best fit their needs. Your skill set includes both basic and custom actions:\n\n1. Basic Actions\nBasic actions are standardized and available across all platforms. They provide essential functionality and are defined with a specific format, ensuring consistency and reliability. \nBasic Action 1: CLICK \n    - purpose: Click at the specified position.\n    - format: CLICK <point>[[x-axis, y-axis]]</point>\n    - example usage: CLICK <point>[[101, 872]]</point>\n       \nBasic Action 2: TYPE\n    - purpose: Enter specified text at the designated location.\n    - format: TYPE [input text]\n    - example usage: TYPE [Shanghai shopping mall]\n\nBasic Action 3: SCROLL\n    - purpose: Scroll in the specified direction.\n    - format: SCROLL [direction (UP/DOWN/LEFT/RIGHT)]\n    - example usage: SCROLL [UP]\n    \n2.Custom Actions\nCustom actions are unique to each user's platform and environment. They allow for flexibility and adaptability, enabling the model to support new and unseen actions defined by users. These actions extend the functionality of the basic set, making the model more versatile and capable of handling specific tasks.\n\n\nCustom Action 1: LONG_PRESS \n    - purpose: Long press at the specified position.\n    - format: LONG_PRESS <point>[[x-axis, y-axis]]</point>\n    - example usage: LONG_PRESS <point>[[101, 872]]</point>\n\nCustom Action 2: PRESS_BACK\n    - purpose: Press a back button to navigate to the previous screen.\n    - format: PRESS_BACK\n    - example usage: PRESS_BACK\n\nCustom Action 3: PRESS_HOME\n    - purpose: Press a home button to navigate to the home page.\n    - format: PRESS_HOME\n    - example usage: PRESS_HOME\n\nCustom Action 4: PRESS_RECENT\n    - purpose: Press the recent button to view or switch between recently used applications.\n    - format: PRESS_RECENT\n    - example usage: PRESS_RECENT\n\nCustom Action 5: IMPOSSIBLE\n    - purpose: Wait for the screen to load.\n    - format: WAIT\n    - example usage: WAIT\n\nCustom Action 6: COMPLETE\n    - purpose: Indicate the task is finished.\n    - format: COMPLETE\n    - example usage: COMPLETE\n\n\nIn most cases, task instructions are high-level and abstract. Carefully read the instruction and action history, then perform reasoning to determine the most appropriate next action. Ensure you strictly generate two sections: Thoughts and Actions.\nThoughts: Clearly outline your reasoning process for current step.\nActions: Specify the actual actions you will take based on your reasoning.\n\nYour current task instruction, action history, and associated screenshot are as follows:\nScreenshot: """
        text2=f"""\nTask: {instruction}\nHistory: \nNone\n"""
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {"type": "image","image": image_path},
                    {"type": "text", "text": text2}
                ],
            }
        ]

        text_prompt = _tokenizer.apply_chat_template(conversation, tokenize=False,add_generation_prompt=True)
        print(text_prompt)
        # Excepted output: '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n'
        image_inputs, video_inputs = process_vision_info(conversation)
        inputs = _tokenizer(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        #inputs = inputs.to("cuda")
        device = _llm.device
        inputs = inputs.to(device)

        # Inference: Generation of the output
        output_ids = _llm.generate(**inputs, max_new_tokens=128)
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]
        output_text = _tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print(output_text[0])

        
        episode["pred"] = os_atlas_2minicpm(output_text[0],use_low_instruction)
        print(episode["pred"])
    except Exception as e:
        print(f"Error: {e}")
        episode["pred"] = NO_THOUGHT_EXAMPLE
    return episode
def run_episode_low(episode, image_path,history_list,use_low_instruction):
    #print(msg)
    #print(episode)
    try:
        global _llm,_tokenizer
        torch.cuda.empty_cache()
        # msg[0]["content"].append(img)
        instruction = episode["instruction"]
        low_instruction = episode["low_instruction"]
        #thought = "Thought: "+low_instruction+"\nAction:"
        history = build_history_actions_str(history_list)
        #history = ""
        a11y_tree = ""
        text = f"""\nYou are a foundational action model capable of automating tasks across various digital environments, including desktop systems like Windows, macOS, and Linux, as well as mobile platforms such as Android and iOS. You also excel in web browser environments. You will interact with digital devices in a human-like manner: by reading screenshots, analyzing them, and taking appropriate actions.\n\nYour expertise covers two types of digital tasks:\n    - Grounding: Given a screenshot and a description, you assist users in locating elements mentioned. Sometimes, you must infer which elements best fit the description when they aren't explicitly stated.\n    - Executable Language Grounding: With a screenshot and task instruction, your goal is to determine the executable actions needed to complete the task.\n\n\nYou are now operating in Executable Language Grounding mode. Your goal is to help users accomplish tasks by suggesting executable actions that best fit their needs. Your skill set includes both basic and custom actions:\n\n1. Basic Actions\nBasic actions are standardized and available across all platforms. They provide essential functionality and are defined with a specific format, ensuring consistency and reliability. \nBasic Action 1: CLICK \n    - purpose: Click at the specified position.\n    - format: CLICK <point>[[x-axis, y-axis]]</point>\n    - example usage: CLICK <point>[[101, 872]]</point>\n       \nBasic Action 2: TYPE\n    - purpose: Enter specified text at the designated location.\n    - format: TYPE [input text]\n    - example usage: TYPE [Shanghai shopping mall]\n\nBasic Action 3: SCROLL\n    - purpose: Scroll in the specified direction.\n    - format: SCROLL [direction (UP/DOWN/LEFT/RIGHT)]\n    - example usage: SCROLL [UP]\n    \n2.Custom Actions\nCustom actions are unique to each user's platform and environment. They allow for flexibility and adaptability, enabling the model to support new and unseen actions defined by users. These actions extend the functionality of the basic set, making the model more versatile and capable of handling specific tasks.\n\n\nCustom Action 1: LONG_PRESS \n    - purpose: Long press at the specified position.\n    - format: LONG_PRESS <point>[[x-axis, y-axis]]</point>\n    - example usage: LONG_PRESS <point>[[101, 872]]</point>\n\nCustom Action 2: PRESS_BACK\n    - purpose: Press a back button to navigate to the previous screen.\n    - format: PRESS_BACK\n    - example usage: PRESS_BACK\n\nCustom Action 3: PRESS_HOME\n    - purpose: Press a home button to navigate to the home page.\n    - format: PRESS_HOME\n    - example usage: PRESS_HOME\n\nCustom Action 4: PRESS_RECENT\n    - purpose: Press the recent button to view or switch between recently used applications.\n    - format: PRESS_RECENT\n    - example usage: PRESS_RECENT\n\nCustom Action 5: IMPOSSIBLE\n    - purpose: Wait for the screen to load.\n    - format: WAIT\n    - example usage: WAIT\n\nCustom Action 6: COMPLETE\n    - purpose: Indicate the task is finished.\n    - format: COMPLETE\n    - example usage: COMPLETE\n\n\nIn most cases, task instructions are high-level and abstract. Carefully read the instruction and action history, then perform reasoning to determine the most appropriate next action. Ensure you strictly generate two sections: Thoughts and Actions.\nThoughts: Clearly outline your reasoning process for current step.\nActions: Specify the actual actions you will take based on your reasoning.\n\nYour current task instruction, action history, and associated screenshot are as follows:\nScreenshot: """
        text2=f"""\nTask: {instruction} You need to: {low_instruction}\nHistory: \n{history}\n"""
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {"type": "image","image": image_path},
                    {"type": "text", "text": text2}
                ],
            }
        ]

        text_prompt = _tokenizer.apply_chat_template(conversation, tokenize=False,add_generation_prompt=True)
        print(text_prompt)
        # Excepted output: '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n'
        image_inputs, video_inputs = process_vision_info(conversation)
        inputs = _tokenizer(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        #inputs = inputs.to("cuda")
        device = _llm.device
        inputs = inputs.to(device)

        # Inference: Generation of the output
        output_ids = _llm.generate(**inputs, max_new_tokens=128)
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]
        output_text = _tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print(output_text[0])

        
        episode["pred"] = os_atlas_2minicpm(output_text[0],use_low_instruction)
        print(episode["pred"])
    except Exception as e:
        print(f"Error: {e}")
        episode["pred"] = NO_THOUGHT_EXAMPLE
    return episode
def os_atlas_2minicpm(action_str,use_low_instruction):
    """
    Convert a string containing low-level thinking and action information to minicpm schema format
    
    Args:
        action_str (str): String containing low-level thinking and action information
        
    Returns:
        dict: Action dictionary in new format
    """
    result = {"STATUS": "continue"}
    
    try:
        # 提取动作部分
        action_start = action_str.find("Actions:")
        action_content = action_str[action_start + len("Actions:"):].strip()
        if action_start == -1:
            action_start = action_str.find("actions:")
            action_content = action_str[action_start + len("actions:"):].strip()
        if action_start == -1:
            raise ValueError("Cannot find action information")
        
        action_content = action_str[action_start + len("Actions:"):].strip()
        
        if "CLICK" in action_content:
            # Extract coordinates
            start = action_content.find("[[") + 2
            end = action_content.find("]]")
            coords_str = action_content[start:end]
            x, y = map(int, coords_str.split(","))
            result["POINT"] = [x, y]
        
        elif "TYPE" in action_content:
            # Extract input text
            start = action_content.find("[") + 1
            end = action_content.find("]")
            text = action_content[start:end]
            result["TYPE"] = text
        
        elif "SCROLL" in action_content:
            # Extract scroll direction
            start = action_content.find("[") + 1
            end = action_content.find("]")
            direction = action_content[start:end]
            direction = direction.strip().lower()
             # If has low instruction, need to reverse direction
            if use_low_instruction:
                if direction == "UP":
                    direction = "DOWN"
                elif direction == "DOWN":
                    direction = "UP"
                elif direction == "LEFT":
                    direction = "RIGHT"
                elif direction == "RIGHT":
                    direction = "LEFT"
            result["to"] = direction
            result["POINT"] = [500, 500]  # 屏幕中心点
        
        elif "LONG_PRESS" in action_content:
            # Extract coordinates
            start = action_content.find("[[") + 2
            end = action_content.find("]]")
            coords_str = action_content[start:end]
            x, y = map(int, coords_str.split(","))
            result["POINT"] = [x, y]
            result["duration"] = 1000  # Default long press duration
        
        elif "PRESS_BACK" in action_content:
            result["PRESS"] = "BACK"
        
        elif "PRESS_HOME" in action_content:
            result["PRESS"] = "HOME"
        
        elif "PRESS_RECENT" in action_content:
            result["PRESS"] = "RECENT"
        
        elif "WAIT" in action_content:
            result["duration"] = 200
        
        elif "COMPLETE" in action_content:
            result["STATUS"] = "finish"
        
        else:
            print(f"Error, invalid action: {action_content}")
            
    except Exception as e:
        print(f"Error: {e}")
        
    return result



def run_episode(episode, image_path,history_list,use_low_instruction):
    if use_low_instruction:
        return run_episode_low(episode, image_path,history_list,use_low_instruction)
    else:
        return run_episode_high(episode, image_path,history_list,use_low_instruction)
def load_image(episode,image_path,history_list,use_low_instruction):
    
    return (episode,image_path,history_list,use_low_instruction)

def predict(args, datasets):
    # set global variable
    global USE_LOW_INSTRUCTION
    USE_LOW_INSTRUCTION = (args.data_name == 'android_control_low_test')
    data_dir = args.data_dir
    split_type = args.split
    print("Predicting on:",datasets)
    
    
    if multiprocessing.get_start_method(allow_none=True) != "spawn":
        multiprocessing.set_start_method("spawn", force=True)
    
    with ProcessPoolExecutor(max_workers=len(DEVICES),initializer=_init_llm,initargs=(args.model_path,)) as poolexec:
        tasks = []
        print("Moving model to devices")
        for device in DEVICES:
            tasks.append(poolexec.submit(move_to, device))
        for t in tasks:
            print(t.result())
    
        for dataset in datasets:
            save_dir = os.path.join(args.output_dir, dataset)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                
            episode_dir = os.path.join(data_dir, split_type, dataset)

            # Use predict.jsonl file to store results (write line by line)
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
                        episode_history = []  # Create a separate history for each episode
                        for prev_episode in episodes[:episode['step_id']-1]:  # Only get history before current step
                            image_path = os.path.join(episode_dir, episodes_file, f"{episodes_file}_{prev_episode['step_id']}.jpeg")
                            if not os.path.exists(image_path):
                                image_path = image_path.replace(".jpeg", ".png")
                            if not os.path.exists(image_path):
                                image_path = episode["image_path"]
                            histroy_action = {
                                "result_action_type": prev_episode['result_action_type'],
                                "result_action_text": prev_episode['result_action_text'],
                                "result_touch_yx": prev_episode['result_touch_yx'],
                                "result_lift_yx": prev_episode['result_lift_yx'],
                                "low_instruction": prev_episode.get("low_instruction",""),
                                "image_path": image_path,
                                "result_action_app_name": prev_episode.get('result_action_app_name', ''),
                            }
                            episode_history.append(histroy_action)
                        episode["category"] = dataset
                        image_path = os.path.join(episode_dir, episodes_file, f"{episodes_file}_{episode['step_id']}.jpeg")
                        if not os.path.exists(image_path):
                            image_path = image_path.replace(".jpeg", ".png")
                        if not os.path.exists(image_path):
                            image_path = episode["image_path"]
                        episode_copy = copy.deepcopy(episode)
                        episode_history_copy = copy.deepcopy(episode_history)
                        future.append(executor.submit(load_image, episode_copy, image_path, episode_history_copy, USE_LOW_INSTRUCTION))

                for f in as_completed(future):
                    all_tasks.append(f.result())

            with open(output_file, "w", encoding="utf-8") as f_out:
                print("Predicting")
                tasks = []
                for task_value in all_tasks:
                    tasks.append(poolexec.submit(run_episode, *task_value))
                
                for task in tqdm(as_completed(tasks), total=len(tasks), dynamic_ncols=True):
                    try:
                        episode = task.result()
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
    parser = argparse.ArgumentParser(description="OS-Atlas Inference")
    parser.add_argument("--seed", type=int, default=2020, help="Random seed")
    parser.add_argument("--model_path", type=str, default=os.getenv("MODEL_NAME", "/home/test/test03/models/OS-Atlas-Pro-7B"),
                       help="Model path")
    parser.add_argument("--output_dir", type=str, 
                       default=os.path.join(os.getenv('OUTPUT_PATH', "eval_results")),
                       help="Directory to save results")
    parser.add_argument("--data_name", type=str, default=os.getenv("PREDICT_DATASET", "chinese_app_test"),
                       help="Eval dataset name")
    args = parser.parse_args()
    random.seed(args.seed)

    # Get dataset information
    args.data_dir, args.split, data_subset = get_dataset_dir(args.data_name)
    
    # Update output directory with model name
    # model_name = args.model_path.split("/")[-2:]  # Get last two parts of model path
    # args.output_dir = os.path.join(args.output_dir, *model_name, args.data_name)
    
    print(f'Loading model at : {args.model_path}')
    print(f'Loading data at  : {args.data_dir}')
    print(f'Processing subsets: {data_subset}')
    print(f'Saving results at: {args.output_dir}')
    
    predict(args, data_subset)
