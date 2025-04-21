import os.path as osp
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
from icecream import ic
import math
import argparse

from PIL import Image, ImageDraw, ImageFont, ImageColor
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_agent.llm.fncall_prompts.nous_fncall_prompt import (
    NousFnCallPrompt,
    Message,
    ContentItem,
)
from utils.evaluator import get_direction
from qwen_vl_utils import smart_resize
import json
from utils.action_utils import *
from utils.utils_qwen.agent_function_call import MobileUse
from IPython.display import display
import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "4"  # 只使用第块G5PU
args = type('Args', (), {})
import torch
torch.manual_seed(1)
def aitw_2_uitars(aitw_action: dict):
    """
    将AITW的动作转换为UITARS的动作格式
    """
    ex_action_type = aitw_action['result_action_type']

    if ex_action_type == ActionType.DUAL_POINT:
        lift_yx = json.loads(aitw_action['result_lift_yx'])
        touch_yx = json.loads(aitw_action['result_touch_yx'])
        if is_tap_action(np.array(touch_yx), np.array(lift_yx)):
            # 点击动作
            click_y, click_x = lift_yx[0], lift_yx[1]
            click_x = int(click_x* 1000)
            click_y = int(click_y* 1000)
            return f"click(start_box=\'<|box_start|>({click_x},{click_y})<|box_end|>\')"
        else:
            # 滑动动作
            touch_yx_new = {
                "x": touch_yx[1],
                "y": touch_yx[0]
            }
            lift_yx_new = {
                "x": lift_yx[1],
                "y": lift_yx[0]
            }
            direction = get_direction(touch_yx_new, lift_yx_new)
            return f"scroll(direction='{direction}')"
    
    elif ex_action_type == ActionType.PRESS_BACK:
        return f"press_back()"
    
    elif ex_action_type == ActionType.PRESS_HOME:
        return f"press_home()"
    
    elif ex_action_type == ActionType.PRESS_ENTER:
        return f"press_enter()"
    elif ex_action_type == ActionType.TYPE:
        return f"type(content='{aitw_action['result_action_text']}')"
    
    elif ex_action_type == ActionType.STATUS_TASK_COMPLETE:
        return f"finished()"
    
    elif ex_action_type == ActionType.STATUS_TASK_IMPOSSIBLE:
        return f"finished()"
    
    elif ex_action_type == ActionType.LONG_POINT:
        lift_yx = json.loads(aitw_action['result_lift_yx'])
        touch_yx = json.loads(aitw_action['result_touch_yx'])
        click_y, click_x = lift_yx[0], lift_yx[1]
        click_x = int(click_x* 1000)
        click_y = int(click_y* 1000)
        return f"long_press(start_box=\'<|box_start|>({click_x},{click_y})<|box_end|>\')"
    elif ex_action_type == ActionType.NO_ACTION:
        return f"wait()"
    elif ex_action_type == ActionType.OPEN_APP:
        return f"open(app_name='{aitw_action['result_action_app_name']}')"
    else:

        print('aitw_action:',aitw_action)
        raise NotImplementedError

    # 返回格式化的JSON字符串
    return json.dumps(qwen_action)
def aitz_2_qwen2_5(aitz_action: dict, resized_height: int, resized_width: int) -> str:
    """
    将AITZ的动作转换为Qwen2.5的动作格式
    
    Args:
        aitz_action (dict): AITZ格式的动作，包含ACTION和ARGS
        resized_height (int): 屏幕高度
        resized_width (int): 屏幕宽度
        
    Returns:
        str: Qwen2.5格式的动作字符串
    """
    aitz_action = json.loads(aitz_action)
    print(aitz_action)
    action_type = aitz_action["ACTION"]
    args = aitz_action["ARGS"]
    
    qwen_action = {}
    
    # 处理点击动作
    if action_type == "CLICK_ELEMENT":
        bbox = args["bbox"]
        # 从bbox [x1, y1, x2, y2] 计算中心点
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        # 将坐标转换为屏幕坐标
        center_x = int(center_x  * resized_width)
        center_y = int(center_y * resized_height)
        qwen_action = {
            "action": "click",
            "coordinate": [center_x, center_y]
        }
    
    # 处理滑动动作
    elif action_type == "SCROLL":
        direction = args["direction"]
        # 根据方向设置起点和终点
        mid_x = resized_width // 2
        mid_y = resized_height // 2
        
        if direction == "up":
            # 从屏幕中间向上滑动（起点在下，终点在上）
            qwen_action = {
                "action": "swipe",
                "coordinate": [mid_x, mid_y + 300],
                "coordinate2": [mid_x, mid_y - 300]
            }
        elif direction == "down":
            # 从屏幕中间向下滑动
            qwen_action = {
                "action": "swipe",
                "coordinate": [mid_x, mid_y - 300],
                "coordinate2": [mid_x, mid_y + 300]
            }
        elif direction == "left":
            # 从屏幕中间向左滑动
            qwen_action = {
                "action": "swipe",
                "coordinate": [mid_x + 300, mid_y],
                "coordinate2": [mid_x - 300, mid_y]
            }
        elif direction == "right":
            # 从屏幕中间向右滑动
            qwen_action = {
                "action": "swipe",
                "coordinate": [mid_x - 300, mid_y],
                "coordinate2": [mid_x + 300, mid_y],
            }
    
    # 处理输入文本
    elif action_type == "INPUT":
        qwen_action = {
            "action": "type",
            "text": args["text"]
        }
    
    # 处理系统按钮
    elif action_type == "PRESS BACK":
        qwen_action = {
            "action": "system_button",
            "button": "Back"
        }
    elif action_type == "PRESS HOME":
        qwen_action = {
            "action": "system_button",
            "button": "Home"
        }
    elif action_type == "PRESS ENTER":
        qwen_action = {
            "action": "system_button",
            "button": "Enter"
        }
    
    # 处理终止动作
    elif action_type == "STOP":
        qwen_action = {
            "action": "terminate",
            "status": args.get("task_status", "success")
        }
    
    # 构建完整的Qwen2.5格式输出
    if qwen_action:
        return f'{{"name":"mobile_use","arguments":{json.dumps(qwen_action)}}}'
    else:
        return ""

def qwen2_5_2_aitz(output_text: str, resized_height: int, resized_width: int) -> str:
    """
    将Qwen2.5的输出转换为AITZ的输出
    """
    action = json.loads(output_text.split('<tool_call>\n')[1].split('\n</tool_call>')[0])
    qwen_action = action['arguments']
    action_name = qwen_action['action']
    # 处理点击动作 long_press这里直接处理成点击 因为没有对应动作
    if action_name == "click" or action_name == "long_press":
        x, y = qwen_action["coordinate"]
        # 将坐标转换为bbox格式 [x1, y1, x2, y2]
        # 这里使用0.1倍屏幕宽度和高度的点击区域
        # 进行归一化
        x = x/ resized_width
        y = y/ resized_height
        return {"ACTION": "CLICK_ELEMENT", "ARGS": {"bbox": [int((x-0.1)*999), int((y-0.1)*999), int((x+0.1)*999), int((y+0.1)*999)]}}
    
    # 处理滑动动作
    elif action_name == "swipe":
        x1, y1 = qwen_action["coordinate"]
        x2, y2 = qwen_action["coordinate2"]
        # hack short swipe and shouble be click （抄的千问的评测逻辑）
        if np.linalg.norm([x2 - x1, y2 - y1]) <= 0.04:
            action_name = "click"
            x1=x1/ resized_width
            y1=y1/ resized_height
            x2=x2/ resized_width
            y2=y2/ resized_height
            return {"ACTION": "CLICK_ELEMENT", "ARGS": {"bbox": [int((x1-0.1)*999), int((y1-0.1)*999), int((x1+0.1)*999), int((y1+0.1)*999)]}}
        # 根据起点和终点判断滑动方向
        if abs(x2 - x1) > abs(y2 - y1):  # 水平滑动
            direction = "right" if x2 > x1 else "left"
        else:  # 垂直滑动
            direction = "down" if y2 > y1 else "up"
        return {"ACTION": "SCROLL", "ARGS": {"direction": direction}}
    
    # 处理输入文本
    elif action_name == "type":
        return {"ACTION": "INPUT", "ARGS": {"text": qwen_action["text"]}}
    
    # 处理系统按钮
    elif action_name == "system_button":
        button = qwen_action["button"]
        if button == "Back":
            return {"ACTION": "PRESS_BACK", "ARGS": {}}
        elif button == "Home":
            return {"ACTION": "PRESS_HOME", "ARGS": {}}
        elif button == "Enter":
            return {"ACTION": "PRESS_ENTER", "ARGS": {}}
    
    # 处理终止动作
    elif action_name == "terminate":
        return {"ACTION": "STOP", "ARGS": {"task_status": qwen_action["status"]}}
    
    # 对于其他动作（如 key,wait,open,long_press等），可能需要忽略或特殊处理
    #key 和open, wait都无法找到对应的action long_press这里直接处理成点击
    return {"ACTION": "", "ARGS": {}}
    


model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
model_path = "/home/test/test03/models/Qwen2.5-VL-7B-Instruct"
user_query_template = 'The user query:{user_request} (You have done the following operation on the current device):'
#model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",device_map="auto")
#processor = AutoProcessor.from_pretrained(model_path)
def get_qwen_response(user_query: str, screenshot: str, args=None, model_path: str = "/home/test/test03/models/Qwen2.5-VL-7B-Instruct") -> tuple:
    """
    获取Qwen模型的响应
    
    Args:
        user_query: 用户查询文本
        screenshot: 截图路径
        model_path: 模型路径，默认使用官方模型
        
    Returns:
        tuple: (response_text, status_code)
    """
    try:
        # 设置默认args
        if args is None:
            args = type('Args', (), {
                'greedy': False,
                'top_p': 0.01,
                'top_k': 1,
                'temperature': 0.01,
                'repetition_penalty': 1.0,
                'presence_penalty': 0.0,
                'out_seq_length': 1024,
                'seed': 1
            })
        
        # 使用args构建参数
        generation_params = {
            'do_sample': not getattr(args, 'greedy', False),
            'top_p': getattr(args, 'top_p', 0.01),
            'top_k': getattr(args, 'top_k', 1),
            'temperature': getattr(args, 'temperature', 0.01),
            'repetition_penalty': getattr(args, 'repetition_penalty', 1.0),
            'presence_penalty': getattr(args, 'presence_penalty', 0.0),
            'max_new_tokens': getattr(args, 'out_seq_length', 1024),
            'seed': getattr(args, 'seed', 1)
        }
        
        # 处理图像尺寸
        dummy_image = Image.open(screenshot)
        #print(dummy_image.size)
        resized_height, resized_width = smart_resize(
            dummy_image.height,
            dummy_image.width,
            factor=processor.image_processor.patch_size * processor.image_processor.merge_size,
            min_pixels=processor.image_processor.min_pixels,
            max_pixels=processor.image_processor.max_pixels,
        )
        #print(resized_height, resized_width)
        # 初始化移动设备接口
        mobile_use = MobileUse(
            cfg={"display_width_px": resized_width, "display_height_px": resized_height}
        )

        # 构建消息
        message = NousFnCallPrompt.preprocess_fncall_messages(
            messages=[
                Message(role="system", content=[ContentItem(text="You are a helpful assistant.")]),
                Message(role="user", content=[
                    ContentItem(text=user_query_template.format(user_request=user_query)),
                    ContentItem(image=f"file://{screenshot}")
                ]),
            ],
            functions=[mobile_use.function],
            lang=None,
        )
        message = [msg.model_dump() for msg in message]
        
        # 处理输入
        text = processor.apply_chat_template(
            message, 
            tokenize=False, 
            add_generation_prompt=True
        )
        print('text:',text)
        inputs = processor(
            text=[text], 
            images=[dummy_image], 
            padding=True, 
            return_tensors="pt"
        ).to('cuda')

        # 如果需要设置随机种子，在generate前设置
        if hasattr(args, 'seed'):
            import torch
            torch.manual_seed(args.seed)

        # 使用正确的参数调用generate
        output_ids = model.generate(
            **inputs, 
            **generation_params
        )
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=True
        )[0]
        aitz_answer=qwen2_5_2_aitz(output_text,resized_height, resized_width)

        return json.dumps(aitz_answer), 200
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return str(e), 500

user_query_template_history = '''The user query: {user_request}
Before answering, explain your reasoning step-by-step in <thinking></thinking> tags, and insert them before the <tool_call></tool_call> XML tags.
After answering, summarize your action in <conclusion></conclusion> tags, and insert them after the <tool_call></tool_call> XML tags.
Task progress (You have done the following operation on the current device):
{history_actions}'''
def aitw_2_qwen2_5_action(aitw_action: dict, resized_height: int, resized_width: int) -> str:
    """
    将AITW的动作转换为Qwen2.5的动作格式
    """
    ex_action_type = aitw_action['result_action_type']
    qwen_action = {"name": "mobile_use", "arguments": {}}

    if ex_action_type == ActionType.DUAL_POINT:
        lift_yx = json.loads(aitw_action['result_lift_yx'])
        touch_yx = json.loads(aitw_action['result_touch_yx'])
        if is_tap_action(np.array(touch_yx), np.array(lift_yx)):
            # 点击动作
            click_y, click_x = lift_yx[0], lift_yx[1]
            click_x = int(click_x* resized_width)
            click_y = int(click_y* resized_height)
            qwen_action["arguments"] = {
                "action": "click",
                "coordinate": [click_x, click_y]
            }
        else:
            # 滑动动作
            qwen_action["arguments"] = {
                "action": "swipe",
                "coordinate": [int(touch_yx[1]* resized_width), int(touch_yx[0]* resized_height)],  # 起点
                "coordinate2": [int(lift_yx[1]* resized_width), int(lift_yx[0]* resized_height)]    # 终点
            }
    
    elif ex_action_type == ActionType.PRESS_BACK:
        button = "Back"
        qwen_action["arguments"] = {
            "action": "system_button",
            "button": button
        }
    
    elif ex_action_type == ActionType.PRESS_HOME:
        button = "Home"
        qwen_action["arguments"] = {
            "action": "system_button",
            "button": button
        }
    elif ex_action_type == ActionType.PRESS_ENTER:
        button = "Enter"
        qwen_action["arguments"] = {
            "action": "system_button",
            "button": button
        }
    elif ex_action_type == ActionType.TYPE:
        qwen_action["arguments"] = {
            "action": "type",
            "text": aitw_action['result_action_text']
        }
    
    elif ex_action_type == ActionType.STATUS_TASK_COMPLETE:
        qwen_action["arguments"] = {
            "action": "terminate",
            "status": "success"
        }
    
    elif ex_action_type == ActionType.STATUS_TASK_IMPOSSIBLE:
        qwen_action["arguments"] = {
            "action": "terminate",
            "status": "failure"
        }
    elif ex_action_type == ActionType.LONG_POINT:
        qwen_action["arguments"] = {
            "action": "long_press",
            "coordinate": [int(aitw_action['result_touch_yx'][1]* resized_width), int(aitw_action['result_touch_yx'][0]* resized_height)],
            "time": 2
        }
    elif ex_action_type == ActionType.NO_ACTION:
        qwen_action["arguments"] = {
            "action": "wait",
            "time": 2
        }
    else:
        print('aitw_action:',aitw_action)
        raise NotImplementedError

    # 返回格式化的JSON字符串
    return json.dumps(qwen_action)
def aitw_2_qwen2_5(aitw_action: dict, resized_height: int, resized_width: int) -> str:
    """
    将AITW的动作转换为Qwen2.5的prompt
    
    """
    aitw_action = json.loads(aitw_action)
    action=aitw_2_qwen2_5_action(aitw_action,resized_height, resized_width)
    thinking = f"<thinking>\n{aitw_action['coat_action_think']}\n</thinking>\n"
    action = f"<tool_call>\n{action}\n</tool_call>\n"
    result = f'<conclusion>\n"{aitw_action["coat_action_desc"]}"\n</conclusion>'
    return thinking + action + result
def get_qwen_response_history(user_query: str, screenshot: str, history_actions: list, model_path: str = "/home/test/test03/models/Qwen2.5-VL-7B-Instruct") -> tuple:
    """
    获取Qwen模型的响应
    
    Args:
        user_query: 用户查询文本
        screenshot: 截图路径
        history_actions: 历史动作
        model_path: 模型路径，默认使用官方模型
        
    Returns:
        tuple: (response_text, status_code)
    """
    #try:
    print('history_actions:',history_actions)
    # 处理图像尺寸
    dummy_image = Image.open(screenshot)
    #print(dummy_image.size)
    resized_height, resized_width = smart_resize(
        dummy_image.height,
        dummy_image.width,
        factor=processor.image_processor.patch_size * processor.image_processor.merge_size,
        min_pixels=processor.image_processor.min_pixels,
        max_pixels=processor.image_processor.max_pixels,
    )
    #print('max_pixels:',processor.image_processor.max_pixels)
    #为12845056 超过4096*3112（4k分辨率） 对于手机分辨率应该足够了
    #将history_actions转换为Qwen2.5的格式
    if history_actions:
        history_actions_str = "".join([f"Step {i+1}: {aitw_2_qwen2_5(action,resized_height, resized_width).replace('<tool_call>','').replace('</tool_call>','').strip()}; " for i, action in enumerate(history_actions)])
    else:
        history_actions_str = ""

    #print(resized_height, resized_width)
    # 初始化移动设备接口
    mobile_use = MobileUse(
        cfg={"display_width_px": resized_width, "display_height_px": resized_height}
    )

    # 构建消息
    message = NousFnCallPrompt.preprocess_fncall_messages(
        messages=[
            Message(role="system", content=[ContentItem(text="You are a helpful assistant.")]),
            Message(role="user", content=[
                ContentItem(text=user_query_template_history.format(user_request=user_query,history_actions=history_actions_str)),
                ContentItem(image=f"file://{screenshot}")
            ]),
        ],
        functions=[mobile_use.function],
        lang=None,
    )
    message = [msg.model_dump() for msg in message]
    
    # 处理输入
    text = processor.apply_chat_template(
        message, 
        tokenize=False, 
        add_generation_prompt=True
    )
    print('text:',text)
    inputs = processor(
        text=[text], 
        images=[dummy_image], 
        padding=True, 
        return_tensors="pt"
    ).to('cuda')

    # 修改generation_params的定义
    generation_params = {
        # 'greedy' 替换为 'do_sample'
        'do_sample': not getattr(args, 'greedy', False),
        'top_p': getattr(args, 'top_p', 0.01),
        'top_k': getattr(args, 'top_k', 1),
        'temperature': getattr(args, 'temperature', 0.01),
        'repetition_penalty': getattr(args, 'repetition_penalty', 1.0),
        # 'presence_penalty' 不支持，可以移除
        # 'out_seq_length' 替换为 'max_new_tokens'
        # 'seed' 不直接支持，需要在外部设置
    }

    # 如果需要设置随机种子，在generate前设置


    # 使用正确的参数调用generate
    output_ids = model.generate(
        **inputs, 
        max_new_tokens=getattr(args, 'out_seq_length', 2048),
        **generation_params
    )
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(
        generated_ids, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=True
    )[0]
    aitz_answer=qwen2_5_2_aitz(output_text,resized_height, resized_width)

    return json.dumps(aitz_answer), 200
        
    #except Exception as e:
    #    print(f"Error: {str(e)}")
    #    return str(e), 500
# 使用示例
if __name__ == "__main__":
    user_query = 'Open the file manager app and view the au_uu_SzH3yR2.mp3 file in MUSIC Folder'
    screenshot = "/home/test/test03/fuyikun/CoAT/data-example/GOOGLE_APPS-523638528775825151/GOOGLE_APPS-523638528775825151_0.png"
    response, state = get_qwen_response(user_query, screenshot)
    print(f"Response: {response}")
    print(f"State: {state}")