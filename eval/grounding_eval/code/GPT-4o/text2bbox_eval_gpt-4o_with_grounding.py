import json
from PIL import Image
import base64
import re
from openai import AsyncClient
import os
import asyncio
import traceback

def read_jsonl(file_path):
    """
    
    读取 JSONL 文件并解析为 Python 字典列表
    :param file_path: JSONL 文件路径
    :return: 包含所有 JSON 对象的列表
    """
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def load_image_from_path(image_path):
    """
    从指定路径加载图片
    :param image_path: 图片文件路径
    :return: PIL.Image 对象
    """
    try:
        image = Image.open(image_path)
        return image
    except Exception as e:
        print(f"Error loading image from {image_path}: {e}")
        return None
    
# Function to encode the image
def encode_image(image_path):
    image = Image.open(image_path)
    w, h = image.width, image.height
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
async def call_GPT(item, client, model_name):
    """
    调用Qwen输出function的描述，输出bbox
    """
    system_prompt = '''
    你是一个GUI组件定位的专家，擅长根据组件的功能描述输出对应的位置。你的任务是根据给定的GUI截图和图中某个文本输出与文本最接近的框的编号
    输入：屏幕截图，文本描述
    输出：屏幕截图中框的编号，以<id></id>为格式
    示例输出一：<id>0</id>
    示例输出二：<id>14</id>
    '''

    image_path = item["image"]
    base64_image = encode_image(image_path)
    content = []
    # 动态添加base64_image部分到 content 列表
    content.append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}",
        },
    })
    content.append({
        "type": "text",
        "text": "屏幕上的文本：{}".format(item["text"])
    })

    res = await client.chat.completions.create(
        messages=[
            {
                "role":"system",
                "content": system_prompt
            },
            {
                "role":"user",
                "content": content,
            }
        ],
        model=model_name,
    )

    response = res.choices[0].message.content
    return response.strip("\n").replace(" ","")

async def verify(response, json_list, ground_truth):
    """
    接受模型的字符串输入，判断是否正确
    """
    pattern = r'<id>\d+</id>'
    matches = re.findall(pattern, response)
    # 将输入字符串转换为整数列表
    if matches:
        try:
            match = matches[0]
            bbox_idx = int(match.replace('<id>','').replace('</id>',''))
            pre_bbox = json_list[bbox_idx]["bbox"]
            
            gt_bbox = list(map(float, ground_truth.strip('<>').split(',')))
            gt_x_min = gt_bbox[0]
            gt_x_max = gt_bbox[2]
            gt_y_min = gt_bbox[1]
            gt_y_max = gt_bbox[3]
            print(pre_bbox, [gt_x_min,gt_y_min,gt_x_max,gt_y_max])
            pre_x_min = pre_bbox[0]
            pre_x_max = pre_bbox[2]
            pre_y_min = pre_bbox[1]
            pre_y_max = pre_bbox[3]
            inter_x_min = max(gt_x_min, pre_x_min)
            inter_x_max = min(gt_x_max, pre_x_max)
            inter_y_min = max(gt_y_min, pre_y_min)
            inter_y_max = min(gt_y_max, pre_y_max)

            # 如果两个 bounding boxes 没有交集，交集面积为 0
            if inter_x_min > inter_x_max or inter_y_min > inter_y_max:
                inter_area = 0
            else:
                inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

            # 计算两个 bounding boxes 的并集
            gt_area = (gt_x_max - gt_x_min) * (gt_y_max - gt_y_min)
            pre_area = (pre_x_max - pre_x_min) * (pre_y_max - pre_y_min)
            union_area = gt_area + pre_area - inter_area

            # 计算 IoU
            iou = inter_area / union_area

            # 判断 IoU 是否大于 50%
            if iou > 0.5:
                return 1
            else:
                return 0
        except:
            return 0
    else:
        print("wrong response: {}".format(response))
    return 0

async def process_item_async(item, json_item, client, model_name, semaphore):
    async with semaphore:
        bbox = await call_GPT(item, client, model_name)
        correct = await verify(bbox,json_item ,item["rel_position"])
        return correct

async def main():
    model_name = "gpt-4o"
    total = 0
    correct = 0
    json_data_path = "your/path/to/the/dataset"
    image_data_path = "your/path/to/annotated/image"
    data = read_jsonl(json_data_path)
    client = AsyncClient(api_key="sk-123")
    semaphore = asyncio.Semaphore(8)
    tasks = []
    for item in data:
        try:
            old_image_path = item["image"]
            old_image_dir = '/'.join(old_image_path.split('/')[:-1])
            new_image_path = item["image"].replace(old_image_dir, image_data_path)
            json_path = new_image_path.replace(".jpeg", ".json")
            json_item = json.load(open(json_path))
            task = asyncio.create_task(process_item_async(item, json_item,client, model_name, image_data_path, semaphore))
            tasks.append(task)
        except:
            pass
    
    results = await asyncio.gather(*tasks)
    for result in results:
        correct += result
        total += 1
    print(correct, total, correct / total)

    return 0

if __name__=="__main__":
    asyncio.run(main())