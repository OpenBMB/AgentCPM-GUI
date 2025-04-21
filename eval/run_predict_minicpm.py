import sys
import multiprocessing
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import json
import torch
import random
import jsonschema
from tqdm import tqdm
from transformers import AutoTokenizer,AutoModelForCausalLM
from concurrent.futures import ProcessPoolExecutor,as_completed,ThreadPoolExecutor
from PIL import Image
from utils.utils import get_dataset_dir
import argparse
import logging
import time

DEVICES = [
    "cuda:0", "cuda:1", "cuda:2", "cuda:3",
    "cuda:4","cuda:5", "cuda:6", "cuda:7",
    ]

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)

if current_dir not in sys.path:
    sys.path.append(current_dir)

def compact_json_dumps(obj):
    return json.dumps(obj, indent=None, separators=(",", ":"), ensure_ascii=False)

ACTION_SCHEMA = json.load(open(os.path.join(current_dir, 'utils/schema', 'schema.json'), encoding="utf-8"))
items = list(ACTION_SCHEMA.items())
insert_index = 3
items.insert(insert_index, ("required", ["thought"])) # enable/disable thought by setting it to "required"/"optional"
ACTION_SCHEMA = dict(items)
SYSTEM_PROMPT = f'''# Role
你是一名熟悉安卓系统触屏GUI操作的智能体，将根据用户的问题，分析当前界面的GUI元素和布局，生成相应的操作。

# Task
针对用户问题，根据输入的当前屏幕截图，输出下一步的操作。

# Rule
- 以紧凑JSON格式输出
- 输出操作必须遵循Schema约束

# Schema
{json.dumps(ACTION_SCHEMA, indent=None, ensure_ascii=False, separators=(',', ':'))}'''

EXTRACT_SCHEMA = json.load(open(os.path.join(current_dir, 'utils/schema', 'schema_for_extraction.json'), encoding="utf-8"))


_llm = None
_tokenizer = None

def _init_llm(model_name):
    global _llm,_tokenizer
    if _llm is None:
        _llm = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True,torch_dtype=torch.bfloat16)
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

def move_to(device):
    global _llm,_tokenizer
    if _llm is None:
        raise ValueError("Error, LLM is not initialized.")
    _llm = _llm.to(device)
    if _tokenizer is None:
        raise ValueError("Error, Tokenizer is not initialized.")
    return f"Moved to {device}"


def run_episode(episode, msg,):
    global _llm,_tokenizer
    outputs = _llm.chat(image=None, msgs=msg, system_prompt=SYSTEM_PROMPT, tokenizer=_tokenizer, temperature=0.1,top_p=0.3,n=1,)
    episode["pred"] = extract_and_validate_json(outputs)
    return episode


def extract_and_validate_json(input_string):
    try:
        json_obj = json.loads(input_string)
        jsonschema.validate(json_obj, EXTRACT_SCHEMA)
        return json_obj
    except json.JSONDecodeError as e:
        print("Error, JSON is NOT valid.")
        return input_string
    except Exception as e:
        print(f"Error, JSON is NOT valid according to the schema.{input_string}", e)
        return input_string

def load_image(episode, image_path, data_name):
    # resize the image proportionally so that the longer side is at most 1120
    def __resize__(origin_img):
        resolution = origin_img.size
        w,h = resolution
        max_line_res = 1120
        if max_line_res is not None:
            max_line = max_line_res
            if h > max_line:
                w = int(w * max_line / h)
                h = max_line
            if w > max_line:
                h = int(h * max_line / w)
                w = max_line
        img = origin_img.resize((w,h),resample=Image.Resampling.LANCZOS)
        return img

    image = Image.open(image_path).convert("RGB")
    image = __resize__(image)

    if data_name == 'android_control_low_test':
        query = episode['low_instruction']
    else:
        query = episode['instruction']

    messages = []
    messages.append(
        {
            "role": "user",
            "content": [
                f"<Question>{query}</Question>\n当前屏幕截图：",
                image
            ]
        }
    )
    return (episode,messages)


def predict(args):
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
                        future.append(executor.submit(load_image, episode, image_path, args.data_name))

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

    import sys

    sys.argv = [
        'run_predict_minicpm_multi_turn.py',  # Simulate command line run
        '--model_path', '/home/test/test12/lyx/ARL-TP/output/MiniCPMV-HW-7B-GRPO-1120px-32s-noKL-ARL-fsdp-mt3-hist5-react/checkpoint-120',
        '--output_dir', 'eval_results/multi_turn',
        '--data_name', 'aitz_test', 
    ]


    parser = argparse.ArgumentParser(description="GUI Agent Inference")
    parser.add_argument("--seed", type=int, default=2020, help="Random seed")
    parser.add_argument("--model_path", type=str, required=True, help="Model path")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--data_name", type=str, required=True, choices=['gui_odyssey_test', 'chinese_app_test', 'aitz_test', 'android_control_high_test', 'android_control_low_test'], help="Eval dataset name")
    args = parser.parse_args()
    random.seed(args.seed)

    print(f'Loading model at : {args.model_path}')
    print(f'Saving results at: {args.output_dir}')

    predict(args)