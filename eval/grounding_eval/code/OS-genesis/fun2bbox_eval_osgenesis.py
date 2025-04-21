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
        return base64.b64encode(image_file.read()).decode("utf-8"),w, h
    
async def call_Qwenvl(item, client, model_name):
    """
    调用Qwen输出function的描述，输出bbox
    """
    sys_prompt = '''你是一个GUI组件定位的专家，擅长根据组件的功能描述输出对应的坐标。你的任务是根据给定的GUI截图和图中某个组件的功能描述输出组件的坐标。
    输入：屏幕截图，功能描述
    输出：边界框的坐标，使用click
    示例输出：actions:\n{\"action_type\":\"click\",\"x\":843,\"y\":231}
    '''
    image_path = item["image"]
    base64_image, w, h = encode_image(image_path)
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
        "text": "当前屏幕的尺寸为{}*{}，屏幕上某一组件的功能描述：{}".format(w, h, item["text"])
    })

    res = await client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": sys_prompt,
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

async def verify(response, ground_truth):
    """
    接受模型的字符串输入，判断是否正确
    """
    pattern = r'actions:\n{"action_type":"click","x":.*?,"y":.*?}'
    matches = re.findall(pattern, response)
    # 将输入字符串转换为整数列表
    if matches:
        match = matches[0]
        try:
            bbox_json = json.loads(match.split('\n')[-1])
            bbox = [int(bbox_json["x"]),int(bbox_json["y"])]
            gt_bbox = list(map(int, ground_truth.strip('<>').split(',')))
            # 遍历每个值，检查是否在ground truth对应值的±5范围内
            gt_x_min = gt_bbox[0]
            gt_x_max = gt_bbox[2]
            gt_y_min = gt_bbox[1]
            gt_y_max = gt_bbox[3]
            print(bbox, [gt_x_min,gt_y_min,gt_x_max,gt_y_max])
            if gt_x_min<=bbox[0]<=gt_x_max and gt_y_min<=bbox[1]<=gt_y_max:
                return 1
            else:
                return 0
        except:
            return 0
    
    else:
        print("wrong response: {}".format(response))
    return 0

async def process_item_async(item, client, model_name, semaphore):
    async with semaphore:
        bbox = await call_Qwenvl(item, client, model_name)
        correct = await verify(bbox,item["abs_position"])
        return correct

async def main():
    model_name = "osgenesis"
    total = 0
    correct = 0
    json_data_path = "your/path/to/the/dataset"
    data = read_jsonl(json_data_path)
    client = AsyncClient(api_key="sk-123", base_url='http://localhost:8000/v1/')
    semaphore = asyncio.Semaphore(16)
    tasks = []
    for item in data:
        task = asyncio.create_task(process_item_async(item, client, model_name,semaphore))
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    for result in results:
        correct += result
        total += 1
    print(correct, total, correct/total)
    return 0

if __name__=="__main__":
    asyncio.run(main())