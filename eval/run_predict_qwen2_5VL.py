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
from utils.utils_qwen.agent_function_call import MobileUse
from qwen_agent.llm.fncall_prompts.nous_fncall_prompt import (
    NousFnCallPrompt,
    Message,
    ContentItem,
)
from utils.qwen_mobile_tool import aitw_2_qwen2_5,qwen2_5_2_aitz,aitw_2_qwen2_5_action
import yaml
import time
import torch
from qwen_vl_utils import smart_resize
import random
from yacs.config import CfgNode as CN
import re
import numpy as np
import requests
import jsonschema
from tqdm import tqdm
from transformers import AutoTokenizer,AutoModelForCausalLM
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from utils.qwen_mobile_tool import aitw_2_uitars
from concurrent.futures import ProcessPoolExecutor,as_completed,ThreadPoolExecutor
from PIL import Image
from utils.utils import get_dataset_dir
DEVICES = [
    "cuda:0", 
    "cuda:1", 
    "cuda:2", "cuda:3",
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


ACTION_SCHEMA = json.load(open(os.path.join(current_dir, 'utils','schema' ,'schema.json'), encoding="utf-8"))

ACTION_THOUGHT_SCHEMA = json.load(open(os.path.join(current_dir, 'utils','schema' ,'schema.json'), encoding="utf-8"))


_llm = None
_tokenizer = None

def _init_llm(model_name):
    global _llm,_tokenizer
    if _llm is None:
        _llm = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, trust_remote_code=True,
                                                                  torch_dtype=torch.bfloat16,
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
#for general task we use this template
user_query_template = '''The user query:  {user_request} 
Task progress (You have done the following operation on the current device): {history_actions}'''
#for gui odyssey we use this template
user_query_template_thought = '''The user query: {user_request}
Before answering, explain your reasoning step-by-step in <thinking></thinking> tags, and insert them before the <tool_call></tool_call> XML tags.
After answering, summarize your action in <conclusion></conclusion> tags, and insert them after the <tool_call></tool_call> XML tags.
Task progress (You have done the following operation on the current device):
{history_actions}'''
#for android control low we use this template
user_query_template_low = '''The user query:  {user_request} 
Current step query: {low_instruction}
Task progress (You have done the following operation on the current device): {history_actions}'''
def get_qwen_response_low(user_query: str, screenshot: str, history_actions: list, low_instruction: str, args=None, model_path: str = "/home/test/test03/models/Qwen2.5-VL-7B-Instruct") -> tuple:
    """
    Get the response from the Qwen model
    
    Args:
        user_query: user query text
        screenshot: screenshot path
        model_path: model path, default using official model
        
    Returns:
        tuple: (response_text, status_code)
    """
    global _llm,_tokenizer
    try:

        # process image size
        dummy_image = Image.open(screenshot)
        #print(dummy_image.size)
        resized_height, resized_width = smart_resize(
            dummy_image.height,
            dummy_image.width,
            factor=_tokenizer.image_processor.patch_size * _tokenizer.image_processor.merge_size,
            min_pixels=_tokenizer.image_processor.min_pixels,
            max_pixels=_tokenizer.image_processor.max_pixels,
        )
        #print(resized_height, resized_width)
        # initialize mobile device interface
        mobile_use = MobileUse(
            cfg={"display_width_px": resized_width, "display_height_px": resized_height}
        )
        print(history_actions)
        if history_actions:
            history_actions_str = "".join([f"Step {i+1}: {aitw_2_qwen2_5_action(action,resized_height, resized_width).strip()}; " for i, action in enumerate(history_actions)])
        else:
            history_actions_str = ""

        # build message
        prompt = NousFnCallPrompt()
        message = prompt.preprocess_fncall_messages(
            messages=[
                Message(role="system", content=[ContentItem(text="You are a helpful assistant.")]),
                Message(role="user", content=[
                    ContentItem(text=user_query_template_low.format(user_request=user_query,history_actions=history_actions_str,low_instruction=low_instruction)),
                    ContentItem(image=f"file://{screenshot}")
                ]),
            ],
            functions=[mobile_use.function],
            lang=None,
        )
        message = [msg.model_dump() for msg in message]
        
        # process input
        text = _tokenizer.apply_chat_template(
            message, 
            tokenize=False, 
            add_generation_prompt=True
        )
        print('text:',text)
        inputs = _tokenizer(
            text=[text], 
            images=[dummy_image], 
            padding=True, 
            return_tensors="pt"
        )
        device = _llm.device
        inputs = inputs.to(device)

        generation_params = {
        # 'greedy' is replaced with 'do_sample'
        'do_sample': not getattr(args, 'greedy', False),
        'top_p': getattr(args, 'top_p', 0.01),
        'top_k': getattr(args, 'top_k', 1),
        'temperature': getattr(args, 'temperature', 0.01),
        'repetition_penalty': getattr(args, 'repetition_penalty', 1.0),
        # 'presence_penalty' is not supported, can be removed
        # 'out_seq_length' is replaced with 'max_new_tokens'
        # 'seed' is not directly supported, needs to be set externally
        }

        # call generate with correct parameters
        output_ids = _llm.generate(
            **inputs, 
            max_new_tokens=getattr(args, 'out_seq_length', 2048),
            **generation_params
        )
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = _tokenizer.batch_decode(
            generated_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=True
        )[0]
        print('output_text:',output_text)
        minicpm_answer=qwen2_5_2_minicpm(output_text,resized_height, resized_width)
        print('minicpm_answer:',json.dumps(minicpm_answer))
        return json.dumps(minicpm_answer), 200
        
    except Exception as e:
        import traceback
        print('error:', str(e))
        print('traceback:', traceback.format_exc())
        return str(e), 500
def get_qwen_response(user_query: str, screenshot: str, history_actions: list, args=None,low_instruction:str=None, model_path: str = "/home/test/test03/models/Qwen2.5-VL-7B-Instruct") -> tuple:
    """
    Get the response from the Qwen model
    
    Args:
        user_query: user query text
        screenshot: screenshot path
        model_path: model path, default using official model
        
    Returns:
        tuple: (response_text, status_code)
    """
    global _llm,_tokenizer
    try:

        # process image size
        dummy_image = Image.open(screenshot)
        #print(dummy_image.size)
        resized_height, resized_width = smart_resize(
            dummy_image.height,
            dummy_image.width,
            factor=_tokenizer.image_processor.patch_size * _tokenizer.image_processor.merge_size,
            min_pixels=_tokenizer.image_processor.min_pixels,
            max_pixels=_tokenizer.image_processor.max_pixels,
        )
        #print(resized_height, resized_width)
        # initialize mobile device interface
        mobile_use = MobileUse(
            cfg={"display_width_px": resized_width, "display_height_px": resized_height}
        )
        print(history_actions)
        if history_actions:
            history_actions_str = "".join([f"Step {i+1}: {aitw_2_qwen2_5_action(action,resized_height, resized_width).strip()}; " for i, action in enumerate(history_actions)])
        else:
            history_actions_str = ""

        # build message
        prompt = NousFnCallPrompt()
        message = prompt.preprocess_fncall_messages(
            messages=[
                Message(role="system", content=[ContentItem(text="You are a helpful assistant.")]),
                Message(role="user", content=[
                    ContentItem(text=user_query_template.format(user_request=user_query,history_actions=history_actions_str)),
                    ContentItem(image=f"file://{screenshot}")
                ]),
            ],
            functions=[mobile_use.function],
            lang=None,
        )
        message = [msg.model_dump() for msg in message]

        # process input
        text = _tokenizer.apply_chat_template(
            message, 
            tokenize=False, 
            add_generation_prompt=True
        )

        print('text:',text)
        inputs = _tokenizer(
            text=[text], 
            images=[dummy_image], 
            padding=True, 
            return_tensors="pt"
        )
        device = _llm.device
        inputs = inputs.to(device)

        generation_params = {
        # 'greedy' is replaced with 'do_sample'
        'do_sample': not getattr(args, 'greedy', False),
        'top_p': getattr(args, 'top_p', 0.01),
        'top_k': getattr(args, 'top_k', 1),
        'temperature': getattr(args, 'temperature', 0.01),
        'repetition_penalty': getattr(args, 'repetition_penalty', 1.0),
        # 'presence_penalty' is not supported, can be removed
        # 'out_seq_length' is replaced with 'max_new_tokens'
        # 'seed' is not directly supported, needs to be set externally
        }

        # call generate with correct parameters
        output_ids = _llm.generate(
            **inputs, 
            max_new_tokens=getattr(args, 'out_seq_length', 2048),
            **generation_params
        )
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = _tokenizer.batch_decode(
            generated_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=True
        )[0]
        print('output_text:',output_text)
        minicpm_answer=qwen2_5_2_minicpm(output_text,resized_height, resized_width)
        print('minicpm_answer:',json.dumps(minicpm_answer))
        return json.dumps(minicpm_answer), 200
        
    except Exception as e:
        import traceback
        print('error:', str(e))
        print('traceback:', traceback.format_exc())
        return str(e), 500
def qwen2_5_2_minicpm(output_text: str, resized_height: int, resized_width: int) -> dict:
    """
    Convert Qwen2.5's output to minicpm's output
    """
    action = json.loads(output_text.split('<tool_call>\n')[1].split('\n</tool_call>')[0])
    qwen_action = action['arguments']
    action_name = qwen_action['action']
    # handle click action, long_press is directly processed as click because there is no corresponding action
    if action_name == "click" :
        x, y = qwen_action["coordinate"]

        # normalize
        x = x/ resized_width*1000
        y = y/ resized_height*1000
        return {"POINT": [int(x), int(y)]}
    elif action_name == "long_press":
        x, y = qwen_action["coordinate"]
        x = x/ resized_width*1000
        y = y/ resized_height*1000
        time=qwen_action["time"]
        # convert time to milliseconds
        time = time*1000
        return {"POINT": [int(x), int(y)], "duration": time}
    
    # handle swipe action
    elif action_name == "swipe":
        x1, y1 = qwen_action["coordinate"]
        x2, y2 = qwen_action["coordinate2"]
        x1 = x1/ resized_width*1000
        y1 = y1/ resized_height*1000
        x2 = x2/ resized_width*1000
        y2 = y2/ resized_height*1000
        # determine swipe direction based on start and end points
        if abs(x2 - x1) > abs(y2 - y1):  # horizontal swipe
            direction = "right" if x2 > x1 else "left"
        else:  # vertical swipe
            direction = "down" if y2 > y1 else "up"
        return {"POINT": [int(x1), int(y1)], "to": direction}
    
    # handle input text
    elif action_name == "type":
        return {"TYPE": qwen_action["text"]}
    
    # handle system button
    elif action_name == "system_button":
        button = qwen_action["button"]
        if button == "Back":
            return {"PRESS": "BACK"}
        elif button == "Home":
            return {"PRESS": "HOME"}
        elif button == "Enter":
            return {"PRESS": "ENTER"}
    
    # handle terminate action
    elif action_name == "terminate":
        return {"STATUS": "finish"}
    elif action_name == "wait":
        # convert time to milliseconds
        time = qwen_action["time"]
        time = time*1000    
        return {"duration": time}
    
    # for other actions (such as key,open, etc.), they may need to be ignored or specially processed
    #key wait cannot find corresponding action
    return {}
def run_episode(episode, image_path,history_list,use_low_instruction):
    query = episode["instruction"]
    screenshot = image_path
    if use_low_instruction:
        low_instruction = episode["low_instruction"]
        output_text,status_code = get_qwen_response_low(query, screenshot,history_list,low_instruction)
    else:
        output_text,status_code = get_qwen_response(query, screenshot,history_list)
    episode["pred"] = extract_and_validate_json(output_text)
    return episode


def extract_and_validate_json(input_string):
    # if input_string == "":
        # raise ValueError("Error, empty output.")
    try:
        json_obj = json.loads(input_string)
        # validate JSON data against Schema
        jsonschema.validate(json_obj, ACTION_THOUGHT_SCHEMA)
        return json_obj
    except json.JSONDecodeError as e:
        print("Error, JSON is NOT valid.",input_string,"over")
        return input_string
    except Exception as e:
        print("Error, JSON is NOT valid according to the schema.",input_string,"over")
        return input_string

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
    parser = argparse.ArgumentParser(description="Qwen2.5VL Inference")
    parser.add_argument("--seed", type=int, default=2020, help="Random seed")
    parser.add_argument("--model_path", type=str, default=os.getenv("MODEL_NAME", "/share_data/data1/GUI_eval/Qwen2.5-VL-7B-Instruct"),
                       help="Model path")
    parser.add_argument("--output_dir", type=str, 
                       default=os.path.join(os.getenv('OUTPUT_PATH', "eval_results")),
                       help="Directory to save results")
    parser.add_argument("--data_name", type=str, default=os.getenv("PREDICT_DATASET", "android_control_high_test"),
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
