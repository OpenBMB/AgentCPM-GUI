import json
from PIL import Image
import base64
import re
from openai import AsyncClient
import os
import asyncio

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
    sys_prompt = '''你是一个GUI组件文字识别的专家，擅长根据组件的边界框（bounding box）描述输出对应的文字。你的任务是根据给定的GUI截图和图中某个组件的边界框输出组件的中的文字。
    输入：屏幕截图，边界框的相对坐标，<x_min, y_min, x_max, y_max>的格式表示
    输出：组件中的文本,注意是文字而非坐标！
    示例输出一：可口可乐。
    示例输出二：关注'''
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
        "text": "屏幕上某一组件的边界框：{}".format(item["rel_position"])
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
        max_tokens = 256
    )

    response = res.choices[0].message.content
    return response.strip("\n").replace(" ","")

async def verify(response, ground_truth):
    """
    接受模型的字符串输入，判断是否正确
    """
    if response == ground_truth:
        return 1
    else:
        return 0

async def process_item_async(item, client, model_name, semaphore):
    async with semaphore:
        response = await call_Qwenvl(item, client, model_name)
        correct = await verify(response,item["text"])
        return correct

async def main():
    model_name = "aguivs"
    total = 0
    correct = 0
    json_data_path = "your/path/to/the/dataset"
    data = read_jsonl(json_data_path)
    client = AsyncClient(api_key="sk-123", base_url='http://localhost:8000/v1')
    semaphore = asyncio.Semaphore(16)
    tasks = []
    for item in data:
        task = asyncio.create_task(process_item_async(item, client, model_name,semaphore))
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    for result in results:
        correct += result
        total += 1
    print(correct, total, correct / total)

    return 0

if __name__=="__main__":
    asyncio.run(main())