import base64
import sys
import argparse
import re
import copy
import multiprocessing
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_ALLOW_DEPRECATED_BEAM_SEARCH"]="1"
import json
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
    #"cuda:0", 
    #"cuda:1", 
    #"cuda:2", "cuda:3",
       "cuda:4", "cuda:5", "cuda:6", "cuda:7",
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
    history = []
    
    # Get indices of the last 4 image records
    image_indices = range(max(0, len(history_list) - 4), len(history_list))
    
    for i, step_history in enumerate(history_list):
     # If current index is in the last 4 image records, add the image
        if i in image_indices:
            image_history = {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": step_history["image_path"]
                    }
                ]
            }
            history.append(image_history)
        
        # Add action
        if i in image_indices:
            action = aitw_2_uitars(step_history)
            thought = step_history.get("low_instruction", "")
            text_history = {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": f"Thought: {thought}\nAction: {action}"}
                ]
            }
            history.append(text_history)
    
    return history
def run_episode_low(episode, image_path,history_list,use_low_instruction):
    #print(msg)
    #print(episode)
    try:
        global _llm,_tokenizer
        torch.cuda.empty_cache()
        # msg[0]["content"].append(img)
        instruction = episode["instruction"]
        low_instruction = episode["low_instruction"]
        thought = "Thought: "+low_instruction+"\nAction:"
        history = build_history_actions_str(history_list)
        text = ("You are a GUI agent. You are given a task and your action history, with screenshots. "
                "You need to perform the next action to complete the task. \n\n"
                "## Output Format\n\n"
                "Thought: ...\n"
                "Action: ...\n\n\n"
                "## Action Space\n"
                "click(start_box=\'<|box_start|>(x1,y1)<|box_end|>\')\n"
                "long_press(start_box=\'<|box_start|>(x1,y1)<|box_end|>\', time=\'\')\n"
                "type(content=\'\')\n"
                "scroll(direction=\'down or up or right or left\')\n"
                #"open_app(app_name=\'\')\n"
                "press_back()\n"
                "press_home()\n"
                "wait()\n"
                "finished() # Submit the task regardless of whether it succeeds or fails.\n\n"
                "## Note\n"
                "- Use English in Thought part.\n\n"
                "- Summarize your next action (with its target element) in one sentence in Thought part.\n\n"
                "## User Instruction\n" + instruction)
        conversation = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text}
                ],
            }
        ]
        conversation.extend(history)
        conversation.append({
            "role": "user",
            "content": [
                {"type": "image", "image": image_path}
            ],
        })
        conversation.append({
            "role": "assistant",
            "content": [
                {"type": "text", "text": thought}
            ],
        })

        text_prompt = _tokenizer.apply_chat_template(conversation, tokenize=False,add_generation_prompt=False)
        # remove <|im_end|> to ensure the continuity of thought and action
        text_prompt = text_prompt.rsplit("<|im_end|>", 1)[0].strip()
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
        output_ids = _llm.generate(**inputs, max_new_tokens=128,temperature=0.1)
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]
        output_text = _tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print(output_text[0])

        
        episode["pred"] = uitars2minicpm(output_text[0])
        print(episode["pred"])
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        print('traceback:', traceback.format_exc())
        episode["pred"] = NO_THOUGHT_EXAMPLE
    return episode
def run_episode_high(episode, image_path,history_list,use_low_instruction):
    #print(msg)
    #print(episode)
    try:
        global _llm,_tokenizer
        torch.cuda.empty_cache()
        # msg[0]["content"].append(img)
        instruction = episode["instruction"]
        #low_instruction = episode["low_instruction"]
        history = build_history_actions_str(history_list)
        text = ("You are a GUI agent. You are given a task and your action history, with screenshots. "
                "You need to perform the next action to complete the task. \n\n"
                "## Output Format\n\n"
                "Thought: ...\n"
                "Action: ...\n\n\n"
                "## Action Space\n"
                "click(start_box=\'<|box_start|>(x1,y1)<|box_end|>\')\n"
                "long_press(start_box=\'<|box_start|>(x1,y1)<|box_end|>\', time=\'\')\n"
                "type(content=\'\')\n"
                "scroll(direction=\'down or up or right or left\')\n"
                #"open_app(app_name=\'\')\n"
                "press_back()\n"
                "press_home()\n"
                "wait()\n"
                "finished() # Submit the task regardless of whether it succeeds or fails.\n\n"
                "## Note\n"
                "- Use English in Thought part.\n\n"
                "- Summarize your next action (with its target element) in one sentence in Thought part.\n\n"
                "## User Instruction\n" + instruction)
        conversation = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text}
                ],
            }
        ]
        conversation.extend(history)
        conversation.append({
            "role": "user",
            "content": [
                {"type": "image", "image": image_path}
            ],
        })

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
        device = _llm.device
        inputs = inputs.to(device)

        # Inference: Generation of the output
        output_ids = _llm.generate(**inputs, max_new_tokens=128,temperature=0.1)
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]
        output_text = _tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        print(output_text[0])

        
        episode["pred"] = uitars2minicpm(output_text[0])
        print(episode["pred"])
    except Exception as e:
        print(f"Error: {e}")
        episode["pred"] = NO_THOUGHT_EXAMPLE
    return episode

def uitars2minicpm(action_str):
    """
    Convert the ui-tars action string to the minicpm schema format
    
    Args:
        action_str (str): like "click(start_box='<|box_start|>(558,925)<|box_end|>')"
        
    Returns:
        dict: new format action dictionary
    """
    result = {"STATUS": "continue"}
    
    # auxiliary function to extract coordinates
    def extract_coords(s):
        # directly find and extract the coordinates in the parentheses
        first_bracket = s.find("(")
        start = s.find("(", first_bracket + 1)
        end = s.find(")")
        if start != -1 and end != -1:
            coords_str = s[start+1:end].strip()  # extract the content in (x,y)
            x, y = coords_str.split(",")
            return [int(x), int(y)]
        raise ValueError(f"Cannot find coordinates in the string: {s}")
    
    if "click(" in action_str:
        result["POINT"] = extract_coords(action_str)
        
    elif "long_press(" in action_str:
        result["POINT"] = extract_coords(action_str)
        if "time='" in action_str:
            time = action_str.split("time='")[1].split("'")[0]
            result["duration"] = int(time) if time else 1000
            
    elif "type(" in action_str:
        content = action_str.split("content='")[1].split("'")[0]
        result["TYPE"] = content
        
    elif "scroll(" in action_str:
        direction = action_str.split("direction='")[1].split("'")[0]
        result["POINT"] = [500, 500]  # screen center point
        #need reverse direction
        if direction == "down":
            direction = "up"
        elif direction == "up":
            direction = "down"
        elif direction == "right":
            direction = "left"
        elif direction == "left":
            direction = "right"
        result["to"] = direction
    elif "press_back()" in action_str:
        result["PRESS"] = "BACK"
        
    elif "press_home()" in action_str:
        result["PRESS"] = "HOME"
        
    elif "wait()" in action_str:
        result["duration"] = 200
        
    elif "finished()" in action_str:
        result["STATUS"] = "finish"
    elif "open_app(app_name=" in action_str:
        result["OPEN_APP"] = action_str.split("app_name='")[1].split("'")[0]
    else:
        print(f"Error, invalid action: {action_str}")
        
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
    parser = argparse.ArgumentParser(description="UI-TARS Inference")
    parser.add_argument("--seed", type=int, default=2020, help="Random seed")
    parser.add_argument("--model_path", type=str, default=os.getenv("MODEL_NAME", "/home/test/test03/models/UI-TARS-7B-SFT"),
                       help="Model path")
    parser.add_argument("--output_dir", type=str, 
                       default=os.path.join(os.getenv('OUTPUT_PATH', "eval_results")),
                       help="Directory to save results")
    parser.add_argument("--data_name", type=str, default=os.getenv("PREDICT_DATASET", "android_control_low_test"),
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
