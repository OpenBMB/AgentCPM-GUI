import base64
import sys
import argparse
import json
import re
import demjson3
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
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from concurrent.futures import ProcessPoolExecutor,as_completed,ThreadPoolExecutor
from PIL import Image
from utils.utils import get_dataset_dir
DEVICES = [
    #"cuda:0", 
    #"cuda:1", 
    #"cuda:2", "cuda:3",
    #   "cuda:4", "cuda:5", 
     "cuda:6", "cuda:7",
    ]
torch.set_num_threads(4)
USE_LOW_INSTRUCTION = False

# Get the absolute path of the current file
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)

# Add the current file's directory to sys.path
if current_dir not in sys.path:
    sys.path.append(current_dir)

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
def run_episode_high(episode, image_path,history_list):
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
        text = f"""You are a GUI task expert, I will provide you with a high-level instruction, an action history, a screenshot with its corresponding accessibility tree.

High-level instruction: {instruction}
Action history: {history}
Accessibility tree: {a11y_tree}

Please generate the low-level thought and action for the next step."""
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image","image": image_path},
                    {"type": "text", "text": text}
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


        episode["pred"] = os_gensis_2minicpm(output_text[0])
        print(episode["pred"])
    except Exception as e:
        print(f"Error: {e}")
        episode["pred"] = NO_THOUGHT_EXAMPLE
    return episode
def run_episode_low(episode, image_path,history_list):
    #print(msg)
    #print(episode)
    try:
        global _llm,_tokenizer
        torch.cuda.empty_cache()
        # msg[0]["content"].append(img)
        instruction = episode["instruction"]
        low_level_thought = episode["low_instruction"]
        #thought = "Thought: "+low_instruction+"\nAction:"
        #history = build_history_actions_str(history_list)
        history = ""
        a11y_tree = ""
        text = f"""You are a GUI task expert, I will provide you with a high-level instruction, an action history,a screenshot with its corresponding accessibility tree, and a low-level thought.
        
        High-level instruction: {instruction}
        Action history: {history}
        Accessibility tree: {a11y_tree}
        Low-level thought: {low_level_thought}
        
        Please generate the low-level thought and action for the next step."""
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image","image": image_path},
                    {"type": "text", "text": text}
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

        
        episode["pred"] = os_gensis_2minicpm(output_text[0])
        print(episode["pred"])
    except Exception as e:
        print(f"Error: {e}")
        episode["pred"] = NO_THOUGHT_EXAMPLE
    return episode
def os_gensis_2minicpm(action_str):
    """
    Convert a string containing low-level thoughts and action information into minicpm schema format

    Args:
        action_str (str): String containing low-level thoughts and action information

    Returns:
        dict: Action dictionary in new format
    """
    result = {"STATUS": "continue"}

    try:
        # 提取动作部分
        action_start = action_str.find("action: ")
        if action_start == -1:
            raise ValueError("Cannot find action information")

        action_json_str = action_str[action_start + len("action: "):].strip()
        print(action_json_str)
        action_dict = demjson3.decode(action_json_str)

        action_type = action_dict.get("action_type")

        if action_type == "type":
            result["TYPE"] = action_dict.get("text", "")
        elif action_type == "click":
            result["POINT"] = [action_dict.get("x", 0), action_dict.get("y", 0)]
        elif action_type == "navigate_home":
            result["PRESS"] = "HOME"

        elif action_type == "navigate_back":
            result["PRESS"] = "BACK"

        elif action_type == "scroll":
            result["POINT"] = [500, 500]  # set default start point
            #if USE_LOW_INSTRUCTION, reverse the direction
            direction = action_dict.get("direction", "down").strip().lower()
            if USE_LOW_INSTRUCTION:
                if direction == "up":
                    direction = "down"
                elif direction == "down":
                    direction = "up"
                elif direction == "left":
                    direction = "right"
                elif direction == "right":
                    direction = "left"
            result["to"] = direction
        elif action_type == "open_app":
            result["OPEN_APP"] = action_dict.get("app_name", "")

        elif action_type == "wait":
            result["duration"] = 200

        elif action_type == "dismiss":
            result["POINT"] = [action_dict.get("x", 0), action_dict.get("y", 0)]

        elif action_type == "long_press":
            result["POINT"] = [action_dict.get("x", 0), action_dict.get("y", 0)]
            result["duration"] = 1000  # set default duration

        elif action_type == "get_text":
            result["POINT"] = [action_dict.get("x", 0), action_dict.get("y", 0)]

        else:
            print(f"Error, invalid action: {action_dict}")

    except json.JSONDecodeError:
        print("Cannot parse action information as JSON")
    except Exception as e:
        print(f"Error: {e}")

    return result

def run_episode(episode, image_path,history_list,use_low_instruction):
    if use_low_instruction:
        return run_episode_low(episode, image_path,history_list)
    else:
        return run_episode_high(episode, image_path,history_list)
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
    parser = argparse.ArgumentParser(description="OS-Genesis Inference")
    parser.add_argument("--seed", type=int, default=2020, help="Random seed")
    parser.add_argument("--model_path", type=str, default=os.getenv("MODEL_NAME", "/share_data/data1/GUI_eval/OS-Genesis-7B-AC"),
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
