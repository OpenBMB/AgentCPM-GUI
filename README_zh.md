<div align="center">

# AgentCPM-GUI 
<!-- 这里是图片的占位符，后续需要补充图片路径 -->
<!--<img src="./assets/.png" width="300em" ></img> -->

**端侧可用的 视觉、语音、多模态GUI Agent**

  <strong>中文 |
  [English](./README.md)</strong>



 <span style="display: inline-flex; align-items: center; margin-right: 2px;">
   <a href="docs/wechat.md" target="_blank"> 微信社区</a> &nbsp;|
 </span>
  <span style="display: inline-flex; align-items: center; margin-left: 2px;">
   MiniCPM-V <a href="docs/best_practice_summary_zh.md" target="_blank">&nbsp; 📖 最佳实践</a>
 </span>
  
  <p align="center">
  AgentCPM-GUI
 <a href="https://huggingface.co/openbmb/MiniCPM-o-2_6">🤗</a> <a href="https://minicpm-omni-webdemo-us.modelbest.cn/"> 🤖</a>
</p>

</div>

**AgentCPM-GUI** 是以MiniCPM-o为基础构建的GUI Agent模型。总参数量8B，接受用户指令，在手机上自动的执行任务。

## 更新日志 <!-- omit in toc -->
* [2025.04.28] 🚀🚀🚀 我们开源了 AgentCPM-GUI，首款端侧GUI Agent大模型，拥有中文APP操作能力，并基于RFT优化思考能力。

## 性能评估

### 定位基准测试

| Model                   | fun2bbox | text2bbox | bbox2text | average |
|-------------------------|----------|-----------|-----------|---------|
| MiniCPM-Agent           | 55.5     | 56.8      | 49.9      | 54.07   |
| Qwen2.5-VL-7B           | 36.8     | 52.0      | 44.1      | 44.30   |
| Qwen2.5-VL-72B          | 68.2     | 76.9      | 59.1      | 68.07   |
| Intern2.5-VL-8B         | 17.2     | 24.2      | 45.9      | 29.10   |
| Intern2.5-VL-26B        | 14.8     | 16.6      | 36.3      | 22.57   |
| GPT-4o                  | 22.1     | 19.9      | 14.3      | 18.77   |
| GPT-4o with Grounding   | 44.3     | 44.0      | 14.3      | 44.15   |


### 智能体基准测试

| Dataset       | Android Control-Low TM | Android Control-Low EM | Android Control-High TM | Android Control-High EM | GUI-Odyssey TM | GUI-Odyssey EM | AITZ TM | AITZ EM | Chinese APP TM | Chinese APP EM |
| ------------- | ---------------------- | ---------------------- | ----------------------- | ----------------------- | -------------- | -------------- | ------- | ------- | -------------- | -------------- |
| **MiniCPM-Agent** | **94.39** | **90.20** | **77.70** | **69.17** | **90.85** | **74.96** | **85.71** | **76.38** | **96.86** | **91.28** |
|Qwen2.5-VL-7B  |92.11|82.12|69.65|57.36|55.33|40.90|73.16|57.58|68.53|48.80|
|UI-TARS-7B     |93.52|88.89|68.53|60.81|78.79|57.33|71.74|55.31|71.01|53.92|
|OS-Genesis     |90.74|74.22|65.92|44.43|11.67|3.63|19.98|8.45|38.10|14.50|
|OS-Atlas       |73.03|67.25|70.36|56.53|91.83|76.76|74.13|58.45|81.53|55.89|
|Aguvis         |93.85|89.40|65.56|54.18|TBD|TBD|35.71|18.99|67.43|38.20|
|OdysseyAgent   |65.10|39.16|58.80|32.74|TBD|73.78|59.17|31.60|67.56|25.44|
|GPT-4o         |-|19.49|-|20.80|-|20.39|70.00|35.30|TBD|TBD|
|Gemini 2.0     |-|28.50|-|60.20|-|3.27|-|-|-|-|
|Claude         |-|19.40|-|12.50|60.90|-|-|-|-|-|

## 推理
```bash
#克隆项目仓库
git clone https://github.com/Zhong-Zhang/MiniCPM-Agent
# 进入项目仓库
cd MiniCPM-Agent
# 下载模型
待写
# 环境配置
pip install -r requirements.txt
```

### 智能体任务

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import json

# 1. 加载模型和分词器
model_path = "model/AgentCPM-GUI"  # 模型路径
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
model = model.to("cuda:0") 

# 2. 构造输入
instruction = "请点击屏幕上的‘设置’按钮"  # 示例指令
image_path = "assets/test.jpg"  # 你的图片路径
image = Image.open(image_path).convert("RGB")

# 3. 构造消息格式
messages = [{
    "role": "user",
    "content": [
        f"<Question>{instruction}</Question>\n当前屏幕截图：",
        image
    ]
}]

# 4. 推理
#这里不太清楚minicpm的system prompt现在是啥样，先随便从路径里导一个进来
with open("my_system_prompt.txt", "r", encoding="utf-8") as f:
    system_prompt = f.read()outputs = model.chat(
    image=None,
    msgs=messages,
    system_prompt=system_prompt,
    tokenizer=tokenizer,
    temperature=0.1,
    top_p=0.3,
    n=1,
)

# 5. 输出结果
print(outputs)
```

### 定位任务



