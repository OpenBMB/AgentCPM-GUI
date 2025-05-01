"""
Inference code is modified from the origin repo: https://github.com/OpenGVLab/GUI-Odyssey
See the origin code at:
Inference: https://github.com/OpenGVLab/GUI-Odyssey/blob/master/src/eval_mm/evaluate_GUIOdyssey.py
Pre-processing: https://github.com/OpenGVLab/GUI-Odyssey/blob/master/data/format_converter.py

Notice: The Odyssey Agent requires a his_index.json to inference. We default put the json file under utils/utils_odyssey.

Please ignore any warnings when running this scripts. The origin repo used an older version, which causes transfer problem.
"""

import re
import os
import sys
import math
import json
import random
import warnings
import argparse
import itertools
from tqdm import tqdm
import multiprocessing
from typing import List, Dict, Literal
from functools import partial
from concurrent.futures import ProcessPoolExecutor,as_completed,ThreadPoolExecutor

Point = Dict[Literal["x","y"], float]
Direction = Literal["up", "down", "left", "right", "no direction"]

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.manual_seed(1234)

from utils.utils import get_dataset_dir
from utils.utils_odyssey.qwen_generation_utils import make_context, decode_tokens
from utils.utils_odyssey.modeling_qwen import QWenLMHeadModel
from utils.utils_odyssey.configuration_qwen import QWenConfig

from utils.utils_odyssey.tokenization_qwen import QWenTokenizer

IMAGE_HISTORY = True

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'utils/utils_odyssey'))

savep:str = "./utils/utils_odyssey/his_index.json"

DEVICES = [
    "cuda:0","cuda:1", "cuda:2", "cuda:3",
    "cuda:4","cuda:5", "cuda:6", "cuda:7",
    ]

_llm = None
_tokenizer = None


def _init_llm(model_name):
    global _llm,_tokenizer
    if _llm is None:
        _llm = QWenLMHeadModel.from_pretrained(model_name, trust_remote_code=True).eval()
    if _tokenizer is None:
        _tokenizer = QWenTokenizer.from_pretrained(model_name, trust_remote_code = True)


def move_to(device):
    global _llm,_tokenizer
    if _llm is None:
        raise ValueError("Error, LLM is not initialized.")
    _llm = _llm.to(device)
    if _tokenizer is None:
        raise ValueError("Error, Tokenizer is not initialized.")
    return f"Moved to {device}"

# ======= Preprocessing: Merge Model
def merge_weight():
    """
    This function is modified from the `merge_weight.py` file of Odyssey repo. It serves as a pre-processing of the model.
    """

    QWENVL_PATH = 'Qwen/Qwen-VL-Chat'
    BACKPACK_DIR = "./utils/utils_odyssey"

    bp_cfg = os.path.join(BACKPACK_DIR, 'config.json')

    tokenizer = AutoTokenizer.from_pretrained(QWENVL_PATH, trust_remote_code=True)
    tokenizer.save_pretrained(BACKPACK_DIR)

    qwen_model = AutoModelForCausalLM.from_pretrained(QWENVL_PATH, device_map=None, trust_remote_code=True)
    cfg = QWenConfig(**json.load(open(bp_cfg)))
    new_qwen_model = QWenLMHeadModel(cfg)

    print("start merging weight...")
    qwen_dict = qwen_model.state_dict()
    odysseyAgent_dict = new_qwen_model.state_dict()
    for k in qwen_dict.keys():
        if k in odysseyAgent_dict:
            odysseyAgent_dict[k] = qwen_dict[k]
    new_qwen_model.load_state_dict(odysseyAgent_dict)
    print("saving...")
    new_qwen_model.save_pretrained(BACKPACK_DIR)

# Util function starts. ========================
def get_direction(point1:Point, point2:Point) -> Direction:
    """
    Util function to get the direction from start point to destination point.

    Args:
        point1(Point), the start point.
        point2(Point), the end point.

    Return:
        str: the direction.
    """
    # Get the coordinate of two points.
    try:
        x1, y1 = point1["x"], point1["y"]
        x2, y2 = point2["x"], point2["y"]

        assert x1 is not None
        assert x2 is not None
        assert y1 is not None
        assert y2 is not None

        vector = (x2 - x1, y2 - y1)
        vx, vy = vector
    except Exception as e:
        return "no direction"

    # Define the direction vector.
    directions = {
        "up": (0, -1),
        "down": (0, 1),
        "left": (-1, 0),
        "right": (1, 0)
    }

    vector_length = math.sqrt(vx ** 2 + vy ** 2)
    if vector_length == 0:  # same point.
        return "no direction"
    unit_vector = (vx / vector_length, vy / vector_length)

    # Calculate the cosine of each direction.
    max_cosine = -float('inf')
    closest_direction = None
    for direction, dir_vector in directions.items():
        dx, dy = dir_vector
        dir_length = math.sqrt(dx ** 2 + dy ** 2)
        cos_theta = (unit_vector[0] * dx + unit_vector[1] * dy) / dir_length
        if cos_theta > max_cosine:
            max_cosine = cos_theta
            closest_direction = direction

    return closest_direction


def transform_actions(action:dict) -> str:
    """
    Transform the action space from minicpm towards Odyssey.

    Args:
        action: the origin action space.

    Returns:
        str: the same action described in odyssey dataset.
    """
    action_type:int = action["result_action_type"]

    if action_type == 7:
        return 'PRESS_ENTER' # Not in Odyssey's action space.
    elif action_type == 1:
        return 'NO_ACTION'   # Not in Odyssey's action space.
    elif action_type == 3:
        return f'TYPE: {action["result_action_text"]}'
    elif action_type == 6:
        return 'PRESS_HOME'
    elif action_type == 5:
        return 'PRESS_BACK'
    elif action_type == 7:
        return 'PRESS_ENTER'
    elif action_type == 10:
        return 'COMPLETE'
    elif action_type == 11:
        return 'IMPOSSIBLE'
    elif action_type == 0:
        # Deal with the long press.
        y, x = map(lambda x: round(x * 1000), json.loads(action["result_touch_yx"]))
        return f'LONG_PRESS ({x}, {y})'

    elif action_type == 4:
        # deal with the click or scroll.
        sy, sx =  map(lambda x: round(x * 1000), json.loads(action["result_touch_yx"]))
        ey, ex =  map(lambda x: round(x * 1000), json.loads(action["result_lift_yx"]))

        if sx == ex and sy == ey:
            return f'CLICK: ({ex}, {ey})'
        else:
            direction = get_direction({"x":sx, "y":sy}, {"x": ex, "y": ey})
            return f'SCROLL: {direction.upper()}'

    else:
        raise NotImplementedError (f"No matching type for type {action_type}")


def make_his_idx(episode_dir:str,
                episodes_files: List[str]):
    """
    This code is used to generate the history index for the Odyssey agent, the origin code could be found in `format_converter.py` file.

    Args:
        episode_dir(str): The directory of the episodes.
        episodes_files(List[str]): The list of episodes files.

    Returns:
        None. The result will be updated directly to the his_index.json file.
    """

    global savep

    his_index: Dict[str,List] = {}

    for episodes_file in episodes_files:
        episodes_path = os.path.join(episode_dir, episodes_file, f"{episodes_file}.json")
        try:
            with open(episodes_path, 'r', encoding='utf-8') as f:
                episodes = json.load(f)
        except Exception as e:
            print(f"Failed to load {episodes_path}: {e}")
            continue

        history_screenshot:List[str] = []

        for episode in episodes:
            image_path = os.path.join(episode_dir, episodes_file, f"{episodes_file}_{episode['step_id']}.jpeg")
            if not os.path.exists(image_path):
                image_path = image_path.replace(".jpeg", ".png")
                if not os.path.exists(image_path):
                    image_path = episode['image_path']

            his_index[image_path] = [cp for cp in history_screenshot]

            history_screenshot.append(image_path)

    try:
        with open(savep, 'r', encoding = "utf-8") as f:
            his_index_old = json.load(f)
    except:
        his_index_old = {}

    his_index_old.update(his_index)

    with open(savep, 'w', encoding='utf-8') as f:
        json.dump(his_index_old, f, ensure_ascii=False, indent=4)


def mapping_actions(action:str) -> dict:
    """
    This functiokn is used to transform the actions from odyssey agent into micicpm action space.

    Args:
        action(str): The action generated from odyssey agent.

    Returns:
        Dict: the action in micicpm action space.
    """

    if action.startswith("CLICK"):
        pattern = r"CLICK: \((\d+),\s*(\d+)\)"
        match = re.match(pattern, action)
        x = int(match.group(1))
        y = int(match.group(2))

        return {
            "POINT": [x, y],
            "duration": 200,
            "STATUS": "continue"
        }

    elif action.startswith("PRESS"):
        sub_action = action.split("_")[-1]

        return {
            "PRESS": sub_action if sub_action != 'RECENT' else 'APPSELECT',
            "duration": 200,
            "STATUS": "continue"
        }

    elif action.startswith("TYPE"):
        text = action.split(":")[-1].strip()

        return {
            "TYPE": text,
            "duration": 200,
            "STATUS": "continue"
        }

    elif action == "COMPLETE":
        return {
            "STATUS": "finish"
        }

    elif action == "IMPOSSIBLE":
        return {
            "STATUS": "impossible"
        }

    elif action.startswith("SCROLL"):
        direction = action.split(":")[-1].strip()

        return {
            "POINT": [500, 500],
            "to": direction.lower(),
            "duration": 200,
            "STATUS": "continue"
        }

    elif action.startswith("LONG_PRESS"):
        pattern = r"LONG_PRESS: \((\d+),\s*(\d+)\)"
        match = re.match(pattern, action)
        x = int(match.group(1))
        y = int(match.group(2))
        return {
            "POINT": [x, y],
            "duration": 1000,
            "STATUS": "continue"
        }

    else:
        print(action)
        raise NotImplementedError

def build_data_episodes(episode_dir:str,
                        episodes_files:List[str],
                        dataset_name:str,
                        his_len:int = 4) -> List[Dict]:
    """
    Transfrom the episode trajectory into Odyssey data.

    This function work as the data preprocess of the origin repo.

    Args:
        episode_dir (str): the directory of the episodes.
        episodes_files (List[str]): the files in the episodes directory.
        dataset_name (str): the name of the dataset.
        his_len (int, optional): the max length of history included. Defaults to 4.

    Returns:
        List[Dict]: a series of data that would be processed later.
    """

    res:list = []

    for episodes_file in episodes_files:
        episodes_path = os.path.join(episode_dir, episodes_file, f"{episodes_file}.json")
        try:
            with open(episodes_path, 'r', encoding='utf-8') as f:
                episodes = json.load(f)
        except Exception as e:
            print(f"Failed to load {episodes_path}: {e}")
            continue
            # Skip this file on error

        history_action:list = []
        history_screenshot:list = []

        for episode in episodes:

            image_path = os.path.join(episode_dir, episodes_file, f"{episodes_file}_{episode['step_id']}.jpeg")
            if not os.path.exists(image_path):
                image_path = image_path.replace(".jpeg", ".png")
                if not os.path.exists(image_path):
                    image_path = episode['image_path']

            gt = transform_actions(episode)

            if dataset_name == 'android_control_high_test':
                instruction = episode["low_instruction"]
            else:
                instruction = episode["instruction"]

            question = f"Picture 1: <img>{image_path}</img>\nI'm looking for guidance on how to {instruction}"

            history_action = history_action[-his_len :]

            if IMAGE_HISTORY:
                if len(history_action) > 0:
                    his_img = f'\nPrevious screenshots: <img>image-history: {image_path}</img>'
                    his_str = '\nPrevious Actions: '
                    for idx, hi in enumerate(history_action):
                        his_str += f"{idx+1}. {hi}\n"

                    question = f"{question}{his_img}{his_str}"
            else:
                if len(history_action) > 0:
                    his_str = '\nPrevious Actions: '
                    for idx, hi in enumerate(history_action):
                        his_str += f"{idx+1}. {hi}\n"

                    question = f"{question}{his_str}"

            res.append(
                {
                    "question": question,
                    'episode': episode
                }
            )

            history_screenshot.append(image_path),
            history_action.append(gt)

    return res

def run_episode(question, episode) -> Dict:

    global _llm, _tokenizer

    raw_text, _ = make_context(
        tokenizer       = _tokenizer,
        query           = question,
        system          = "You are a helpful assistant.",
        max_window_size = 6144,
        chat_format     = "chatml")

    input_ids = _tokenizer(raw_text, return_tensors='pt', padding='longest')

    attention_mask = input_ids.attention_mask
    input_ids = input_ids.input_ids

    input_ids = input_ids.to(_llm.device)
    attention_mask = attention_mask.to(_llm.device)

    out_ids = _llm.generate(
        input_ids            = input_ids,
        attention_mask       = attention_mask,
        do_sample            = False,
        num_beams            = 1,
        length_penalty       = 1,
        num_return_sequences = 1,
        use_cache            = True,
        pad_token_id         = _tokenizer.eod_id,
        eos_token_id         = _tokenizer.eod_id,
        min_new_tokens       = 1,
        max_new_tokens       = 30,
    )

    padding_len = input_ids[0].eq(_tokenizer.pad_token_id).sum().item()
    response = decode_tokens(
        out_ids[0][padding_len: ],
        _tokenizer,
        raw_text_len          = len(raw_text),
        context_length        = input_ids.size(1) - padding_len,
        chat_format           = "chatml",
        verbose               = False,
        errors                = 'replace'
    )

    episode["pred"] = mapping_actions(response)

    return episode


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
        for fut in futures:
            print(fut.result())

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

            all_tasks = []
            print("Loading episodes")

            all_tasks = build_data_episodes(
                episode_dir    = episode_dir,
                episodes_files = episodes_files,
                dataset_name   = args.data_name,
                his_len        = args.his_len
            )

            with open(output_file, "w", encoding="utf-8") as f_out:
                print("Predicting")
                tasks = []
                for task_value in all_tasks:
                    tasks.append(poolexec.submit(run_episode, **task_value))

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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=2020, help="Random seed")
    parser.add_argument("--model_path", type=str, required=True, help="Model path")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--data_name", type=str, required=True, choices=['gui_odyssey_test', 'chinese_app_test', 'aitz_test', 'android_control_high_test', 'android_control_low_test'], help="Eval dataset name")
    parser.add_argument('--batch_size', type=int, default=4, help="Set the batch size of inference process")
    parser.add_argument('--num_workers', type=int, default=12, help="The number of workers in torch distributed.")
    parser.add_argument('--image_history', type=str, default='yes', choices=["yes", "no"], help="Whether to collect the history for inference")
    parser.add_argument('--his_len', type=int, default=4, help="The maximum length of history collected.")

    args = parser.parse_args()

    random.seed(args.seed)

    print(f'Loading model at : {args.model_path}')
    print(f'Saving results at: {args.output_dir}')

    predict(args)

