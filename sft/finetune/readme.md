# Fine-tuning Scripts (Based on MiniCPM-V)

This repository provides fine-tuning scripts adapted from MiniCPM-o, supporting fine-tuning the model on GUI agent tasks. For detailed parameter configurations, please refer to the [MiniCPM-o](https://github.com/OpenBMB/MiniCPM-o/tree/main/finetune) repository.

---

## 📂 Data Preparation

Each training sample should be a dictionary with:

- `id`: a unique identifier,
- `image`: either a single image path (string) or a dictionary of image paths with placeholders (`<image_00>`, `<image_01>`, ...),
- `conversations`: a list of dialogue turns between user and assistant.

#### Single Image Example

```json
[
  {
    "id": "0",
    "image": "path/to/image.jpg",
    "conversations": [
      {"role": "system", "content": "system prompt"},
      {"role": "user", "content": "<image>\nWhat is in the image?"},
      {"role": "assistant", "content": "The image contains..."}
    ]
  }
]
```

#### Multi-Image Example

```json
[
  {
    "id": "0",
    "image": {
      "<image_00>": "path/to/image0.jpg",
      "<image_01>": "path/to/image1.jpg"
    },
    "conversations": [
      {"role": "system", "content": "system prompt"},
      {"role": "user", "content": "Compare the objects.\n<image_00>\n<image_01>"},
      {"role": "assistant", "content": "The first image shows..."}
    ]
  }
]
```

If no image placeholder is present in the text, the image embedding will be prepended by default.

---

## 🚀 Training Setup

#### Full-Parameter Finetuning

Edit and run  "finetune_ds.sh":

```bash
MODEL="openbmb/MiniCPM-V-2_6" # or "openbmb/MiniCPM-o-2_6", openbmb/MiniCPM-Llama3-V-2_5, openbmb/MiniCPM-V-2
DATA="path/to/trainging_data" # json file
EVAL_DATA="path/to/test_data" # json file
LLM_TYPE="qwen" # if use openbmb/MiniCPM-V-2, please set LLM_TYPE=minicpm, if use openbmb/MiniCPM-Llama3-V-2_5, please set LLM_TYPE="llama3",
# if use openbmb/MiniCPM-o-2_6 or openbmb/MiniCPM-V-2_6, please set LLM_TYPE=qwen
```

---

## 📃 Training Data Sample

```json
[{"id":"0","image":{"<image_00>":"/image_path/screenshot.jpeg"},"conversations":[{"role":"system","content":"# Role\n你是一名熟悉安卓系统触屏GUI操作的智能体，将根据用户的问题，分析当前界面的GUI元素和布局，生成相应的操作。\n\n# Task\n针对用户问题，根据输入的当前屏幕截图，输出下一步的操作。\n\n# Rule\n- 以紧凑JSON格式输出\n- 输出操作必须遵循Schema约束\n\n# Schema\n{\"type\":\"object\",\"description\":\"执行操作并决定当前任务状态\",\"additionalProperties\":false,\"optional\":[\"thought\"],\"properties\":{\"thought\":{\"type\":\"string\",\"description\":\"智能体的思维过程\"},\"POINT\":{\"$ref\":\"#/$defs/Location\",\"description\":\"点击屏幕上的指定位置\"},\"to\":{\"description\":\"移动，组合手势参数\",\"oneOf\":[{\"enum\":[\"up\",\"down\",\"left\",\"right\"],\"description\":\"从当前点（POINT）出发，执行滑动手势操作，方向包括向上、向下、向左、向右\"},{\"$ref\":\"#/$defs/Location\",\"description\":\"移动到某个位置\"}]},\"duration\":{\"type\":\"integer\",\"description\":\"动作执行的时间或等待时间，毫秒\",\"minimum\":0,\"default\":200},\"PRESS\":{\"type\":\"string\",\"description\":\"触发特殊按键，HOME为回到主页按钮，BACK为返回按钮，ENTER为回车按钮\",\"enum\":[\"HOME\",\"BACK\",\"ENTER\"]},\"TYPE\":{\"type\":\"string\",\"description\":\"输入文本\"},\"STATUS\":{\"type\":\"string\",\"description\":\"当前任务的状态。特殊情况：satisfied，无需操作；impossible，任务无法完成；interrupt，任务中断；need_feedback，需要用户反馈；\",\"enum\":[\"continue\",\"finish\",\"satisfied\",\"impossible\",\"interrupt\",\"need_feedback\"],\"default\":\"continue\"}},\"$defs\":{\"Location\":{\"type\":\"array\",\"description\":\"坐标为相对于屏幕左上角位原点的相对位置，并且按照宽高比例缩放到0～1000，数组第一个元素为横坐标x，第二个元素为纵坐标y\",\"items\":{\"type\":\"integer\",\"minimum\":0,\"maximum\":1000},\"minItems\":2,\"maxItems\":2}}}"},{"role":"user","content":"<Question>打开美团外卖</Question>\n当前屏幕截图：<image_00>"},{"role":"assistant","content":"{\"POINT\":[197,634],\"to\":\"right\"}"}]}]
```
