# README for Evaluation

Here, we have listed all the evaluation codes. Since each model has different image processing and action spaces, we have organized the evaluation codes by model.

## General Notification

### step 1
Please first download the corresponding images and replace the empty folder `eval/grounding_eval/dataset/images`.

### step 2
We recommend using vLLM as the engine for inference of most open-source models (except InternVL) to ensure inference speed. The specific command is as follows:
```code
python -m vllm.entrypoints.openai.api_server --model /path/to/your/model --served-model-name name_of_your_model --tensor-parallel-size 4
```

### step 3
Modify the `json_data_path` variable in the script to the path of your dataset JSONL file, and then run the script:
```
python your_evaluation_script_name.py
```

## Notification for Special Models

### InternVL series
For the InternVL series of models, we rely on the model loading code provided in their open-source repository for inference. Before running the evaluation code, you need to clone the InternVL open-source repository and install the dependencies. Then, place the inference code under the path `InternVL/internvl_chat/eval`.
After that, run the command:
```
torchrun --nproc_per_node=8 path/to/your/evaluate_script.py --checkpoint ${CHECKPOINT} --dynamic
```

### GPT-4o with grounding
For GPT-4o, in addition to testing its direct grounding capabilities, we also use the Omni-parser to draw bounding boxes on the components in the images, allowing GPT-4o to select the most similar bounding box and then calculate the IoU.

To reproduce this result, you need to first use Omni-parser to process the images, put the script `grounding_eval/GPT-4o/process_image.py` in following position:

```
Omniparser
    ├──docs
    ...
    ├──weights
    ├──utils
    └──process_image.py
```
run the code and save the annotated image and bounding box in `your/path/to/annotated/image`

Finally, modify the `image_data_path` variable in the script to `your/path/to/annotated/image`, and run the code.
