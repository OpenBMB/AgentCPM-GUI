import json
from PIL import Image
import base64
import re
from openai import AsyncClient
import os
import asyncio
import io

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
    image = resize(image)
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")  # 保存为 JPEG 格式
    image_bytes = buffered.getvalue()
    
    # 对字节流进行 Base64 编码
    base64_encoded = base64.b64encode(image_bytes).decode("utf-8")
    return base64_encoded, w, h

def resize(origin_img):
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
    
def process_position(item, w, h):
    pattern = r'<\d+, \d+, \d+, \d+>'
    matches = re.findall(pattern, item["abs_position"])
    if matches:
        match = matches[0]
        bbox = list(map(int, match.strip('<>').split(',')))
        rel_position = [int(bbox[0]/w*1000), int(bbox[1]/h*1000),int(bbox[2]/w*1000),int(bbox[3]/h*1000)]
    print(rel_position)
    return rel_position[0],rel_position[1],rel_position[2],rel_position[3]
    
async def call_Qwenvl(item, client, model_name):
    """
    调用Qwen输出function的描述，输出bbox
    """
    sys_prompt = '''你是一个GUI组件文字识别的专家，擅长根据组件的边界框（bounding box）描述输出对应的文字。你的任务是根据给定的GUI截图和图中某个组件的边界框输出组件的中的文字。\n    输入：屏幕截图，边界框的坐标\n    输出：组件中的文本'''
    image_path = item["image"]
    base64_image, w, h = encode_image(image_path)
    x_min, y_min, x_max, y_max = process_position(item, w, h)
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
        "text": "屏幕上某一组件的边界框：{{\"bbox\":[[{},{}],[{},{}]]\}}".format(x_min,y_min,x_max,y_max)
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
    print(response, ground_truth)
    if response == ground_truth:
        return 1
    
    return 0

async def process_item_async(item, client, model_name, semaphore):
    async with semaphore:
        response = await call_Qwenvl(item, client, model_name)
        correct = await verify(response,item["text"])
        return correct

async def main():
    model_name = "minicpm"
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