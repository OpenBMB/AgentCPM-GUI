import sys
import multiprocessing
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import json
import torch
import random
from jsonschema import Draft7Validator
import jsonschema
from jsonschema.exceptions import ValidationError
from tqdm import tqdm
from transformers import AutoTokenizer,AutoModelForCausalLM
from concurrent.futures import ProcessPoolExecutor,as_completed
from PIL import Image
from utils.utils import get_dataset_dir
import argparse
import re

DEVICES = [
    "cuda:0", 
    "cuda:1", "cuda:2", "cuda:3",
    "cuda:4","cuda:5", "cuda:6", "cuda:7",
    ]

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)

if current_dir not in sys.path:
    sys.path.append(current_dir)

def compact_json_dumps(obj):
    return json.dumps(obj, indent=None, separators=(",", ":"), ensure_ascii=False)

ACTION_SCHEMA = json.load(open(os.path.join(current_dir, 'utils/schema', 'schema.json'), encoding="utf-8"))
ACTION_SCHEMA['properties'].pop('thought')
SYSTEM_PROMPT = """# Role
你是一个智能助手，分析并执行动作以完成用户需求。

# 输出格式
<think>将你的思考过程放用这两个tag括起来</think><act>{...用JSON串表示的动作...}</act>

# 规则
- 你需要在<think>标签中写下你的思考过程
- 你需要在<act>标签中写下你的动作
- 输出的动作必须遵循Schema约束

# Schema
""" + compact_json_dumps(ACTION_SCHEMA)

SYSTEM_PROMPT = """# Role
你是一个智能助手

# 输出格式
你有多种可选的输出格式，按需选择一种即可

# 输出格式1
<plan>...初始或更新后的计划...</plan><think>将你的思考过程放用这两个tag括起来</think><act>{...用紧凑JSON串表示的动作...}</act>

# 输出格式2
<reflection>...对计划的反思...</reflection><plan></plan><think>...</think><act>{...}</act>

# 输出格式3
<think>...</think><act>{...}</act>

# 规则
- 你需要在<think>标签中写下你的思考过程
- 你需要在<act>标签中写下你的动作
- 输出的动作必须遵循Schema约束
- 每次只能输出一个动作
- 当用户提供问题后，在<plan>标签内制定一个执行计划，并在后续执行中更新这个执行计划
- 你的思考内容至少需要包括整体计划，对历史结果的思考和当前状态的分析

## 计划示例
<plan>
[] 思考当前界面，分析用户需求
[] 在xx中...
[] [] 打开...
[] [] 点击...
...
</plan>

# 提示
- 尽可能多样的思考，避免简单的无效思考例如“我需要点击这个按钮”或“我需要滑动”，而是要考虑到当前状态和历史信息的影响
- 对当前状态的分析应该从尽可能多的方面进行，例如当前界面是否符合预期，任务的执行状态，计划是否正常进行等等
- 尽可能完备的考虑历史信息，例如可以从历史信息中发现错误，是否需要回退，是否应该继续或是更新计划
- 你的历史思考过程也已经提供，你需要结合过去的思考和当前状态进行反思，可以围绕计划的执行情况，计划的合理性，可行性等方面进行思考
- 在对上一轮结果的分析中对计划执行情况进行更新
- 当执行结果不符合预期时，考虑计划是否合理，若不合理，需要重新制定计划
- 需要执行滑动操作时，需要注意操作方向和屏幕移动的方向是XY轴镜像的
- 动作有很多种可能性，例如点击，滑动，输入文本，触发特殊按键等。当你不确定应该执行什么动作时，可以考虑在一个JSON串中组合多个动作进行探索: <act>{"to":"up","duration":1000,"PRESS":"BACK","TYPE":"abc","POINT":[123,456]}</act>
- 你需要在思考中给出更多的背景信息，例如“当前界面未找到符合要求的商品，需要向下滑动查看更多商品”或者“当前界面正在加载，请等待”


# 示例
以下是给定的一些简单示例，在正常情况下，你应该提供比以下示例思考更复杂的思考过程

## 示例 1
<think>当前界面未找到符合要求的商品，需要向下滑动查看更多商品</think><act>{"to":"up","POINT":[123,456]}</act>

## 示例 2
<think>界面中显示的内容不符合期望，我应该回退到上个界面重新选择</think><act>{"PRESS":"BACK"}</act>

## 示例 3
<think>当前界面正在加载，请等待</think><act>{"duration":3000}</act>

## 示例 4
<think>当前界面已经完成了任务，我需要结束任务</think><act>{"STATUS":"finish"}</act>

# Schema
""" + compact_json_dumps(ACTION_SCHEMA)

EXTRACT_SCHEMA = json.load(open(os.path.join(current_dir, 'utils/schema', 'schema_for_extraction.json'), encoding="utf-8"))
validator = Draft7Validator(EXTRACT_SCHEMA)

_llm = None
_tokenizer = None

def _init_worker(model_path: str):
    global _llm, _tokenizer
    if _llm is None:
        _llm = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

def move_to(device: str):
    global _llm
    torch.cuda.set_device(device)
    _llm = _llm.to(device)
    return f"Moved to {device}"


def extract_and_validate_json(input_string: str):
    """
    Parse <think> and <act> from model output and validate against EXTRACT_SCHEMA.
    Returns a merged dict on success or the original string on failure.
    """
    # Extract thought
    thought = None
    think_match = re.search(r'<think>(.*?)</think>', input_string, flags=re.S)
    if think_match:
        thought = think_match.group(1).strip()

    # Extract action JSON
    act_match = re.search(r'<act>\s*(\{.*?\})\s*</act>', input_string, flags=re.S)
    if not act_match:
        print("missing <act> tag")
        return input_string
    json_str = act_match.group(1)

    # Parse and validate
    try:
        action = json.loads(json_str)
        validator.validate(action, EXTRACT_SCHEMA)
        # Merge and return
        return {"thought": thought, **action}
    except (json.JSONDecodeError, ValidationError) as e:
        print(f"Error parsing/validating action: {e}")
        return input_string

# def extract_and_validate_json(input_string):
#     try:
#         json_obj = json.loads(input_string)
#         jsonschema.validate(json_obj, EXTRACT_SCHEMA)
#         return json_obj
#     except json.JSONDecodeError as e:
#         print("Error, JSON is NOT valid.")
#         return input_string
#     except Exception as e:
#         print(f"Error, JSON is NOT valid according to the schema.{input_string}", e)
#         return input_string

def _resize(origin_img, max_line_res=1120):
    resolution = origin_img.size
    w,h = resolution
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


def load_step(step, episode_dir, episodes_file, data_name):
    """Return (step_dict, query, img_full1120, img_hist448)."""
    img_path = os.path.join(
        episode_dir, episodes_file, f"{episodes_file}_{step['step_id']}.jpeg")
    if not os.path.exists(img_path):
        img_path = img_path.replace(".jpeg", ".png")
        if not os.path.exists(img_path):
            img_path = step['image_path']

    img = Image.open(img_path).convert("RGB")
    img_full = _resize(img, max_line_res=1120)
    img_hist = _resize(img, max_line_res=448)

    query = step['low_instruction'] if data_name == 'android_control_low_test' else step['instruction']
    return step, query, img_full, img_hist


def run_episode_multi(meta: tuple) -> list:
    episode_dir, ep_file, steps_raw, hist_len, data_name = meta
    steps = [load_step(st, episode_dir, ep_file, data_name) for st in steps_raw]
    assistant_responses, user_msgs_cache, preds = [], [], []
    for cur_idx, (step, query, img_full, img_hist) in enumerate(steps):
        conv = []
        if hist_len > 0:
            for past_idx, raw_out in enumerate(assistant_responses):
                q, past_img_hist = user_msgs_cache[past_idx]
                if cur_idx - past_idx <= hist_len:
                    content = [f"当前屏幕截图：", past_img_hist]
                else:
                    content = f"// 该图像为历史图像，无法显示"
                conv.append({"role": "user", "content": content})
                conv.append({"role": "assistant", "content": raw_out})
        
        conv.append({"role": "user", "content": [f"<Question>{query}</Question>\n当前屏幕截图：", img_full]})
        raw_out = _llm.chat(
            image=None,
            msgs=conv,
            system_prompt=SYSTEM_PROMPT,
            tokenizer=_tokenizer,
            temperature=0.1,
            top_p=0.3,
            n=1,
            max_new_tokens=1024
        )
        step['pred'] = extract_and_validate_json(raw_out)
        step['raw_pred'] = raw_out
        preds.append(step)
        assistant_responses.append(raw_out)
        user_msgs_cache.append((query, img_hist))
    return preds

def predict(args):
    """
    核心预测逻辑，接受解析后 args：
    --model_path, --output_dir, --data_name, --hist_len, --seed
    """
    random.seed(args.seed)
    data_dir, split, data_subset = get_dataset_dir(args.data_name)

    if multiprocessing.get_start_method(allow_none=True) != 'spawn':
        multiprocessing.set_start_method('spawn', force=True)

    os.makedirs(args.output_dir, exist_ok=True)

    with ProcessPoolExecutor(
        max_workers=len(DEVICES),
        initializer=_init_worker,
        initargs=(args.model_path,),
    ) as pool:
        results = [pool.submit(move_to, dev) for dev in DEVICES]
        for fut in results: print(fut.result())

        # 遍历数据子集
        for dataset in data_subset:
            print(f"Predicting {dataset}...")
            root = os.path.join(data_dir, split, dataset)
            if not os.path.isdir(root):
                continue
            save_dir = os.path.join(args.output_dir, dataset)
            os.makedirs(save_dir, exist_ok=True)
            output_file = os.path.join(save_dir, 'predict.jsonl')

            # 构建元数据列表
            traj_data = []
            for ep in os.listdir(root):
                ep_path = os.path.join(root, ep, f"{ep}.json")
                try:
                    steps_raw = json.load(open(ep_path, 'r', encoding='utf-8'))
                except Exception:
                    continue
                steps_raw.sort(key=lambda x: x['step_id'])
                for s in steps_raw:
                    s['category'] = dataset
                traj_data.append((root, ep, steps_raw, args.hist_len, args.data_name))

            # 并行推理并写入文件
            futures = [pool.submit(run_episode_multi, meta) for meta in traj_data]
            with open(output_file, 'w', encoding='utf-8') as fout:
                for fut in tqdm(as_completed(futures), total=len(futures), dynamic_ncols=True):
                    for step in fut.result():
                        try:
                            fout.write(json.dumps(step, ensure_ascii=False) + "\n")
                            fout.flush()
                        except Exception as e:
                            print(f"Error: {e}")
                            continue

            print(f"Saved: {output_file}")

    # 合并所有结果
    os.system(f"cat {args.output_dir}/*/predict.jsonl > {args.output_dir}/all.jsonl")
    print(f"Merged at {args.output_dir}/all.jsonl")


if __name__ == "__main__":

    import sys

    sys.argv = [
        'run_predict_minicpm_multi_turn.py',  # Simulate command line run
        '--model_path', '/share_data/data3/workhome/zhangzhong/checkpoint/checkpoint-330',
        # '--model_path', '/share_data/data3/workhome/zhangzhong/checkpoint/open_app/checkpoint-20763',
        '--output_dir', 'eval_results/multi_turnxxx',
        '--data_name', 'chinese_app_test',
    ]

    parser = argparse.ArgumentParser(description="GUI Agent Inference")
    parser.add_argument("--seed", type=int, default=2020, help="Random seed")
    parser.add_argument("--model_path", type=str, required=True, help="Model path")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--data_name", type=str, required=True, choices=['gui_odyssey_test', 'chinese_app_test', 'aitz_test', 'android_control_high_test', 'android_control_low_test'], help="Eval dataset name")
    parser.add_argument("--hist_len", type=int, default=3, help="History length")
    args = parser.parse_args()
    random.seed(args.seed)

    print(f'Loading model at : {args.model_path}')
    print(f'Saving results at: {args.output_dir}')

    predict(args)
