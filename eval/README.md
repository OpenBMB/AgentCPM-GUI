# Evaluation Scripts

## Grounding Benchmark

| Model                   | fun2bbox | text2bbox | bbox2text | average |
|-------------------------|----------|-----------|-----------|---------|
| MiniCPM-Agent           | 55.5     | 56.8      | 49.9      | 54.07   |
| Qwen2.5-VL-7B           | 36.8     | 52.0      | 44.1      | 44.30   |
| Qwen2.5-VL-72B          | 68.2     | 76.9      | 59.1      | 68.07   |
| Intern2.5-VL-8B         | 17.2     | 24.2      | 45.9      | 29.10   |
| Intern2.5-VL-26B        | 14.8     | 16.6      | 36.3      | 22.57   |
| OS-Genesis-7B           | 8.3      | 5.8       | 4.0       | 6.03    |
| UI-TARS-DPO             | 56.8     | 66.7      | 1.4       | 41.63   |
| OS-Altas-7B             | 53.6     | 60.7      | 0.4       | 38.23   |
| Aguvis                  | 60.8     | 76.5      | 0.2       | 45.83   |
| GPT-4o                  | 22.1     | 19.9      | 14.3      | 18.77   |
| GPT-4o with Grounding   | 44.3     | 44.0      | 14.3      | 44.15   |


## Agent Benchmark

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

## MiniCPM-Agent

### Inference

```
# aitz_test
python run_predict_minicpm.py --model_path /path/to/model --output_dir ./eval_results/aitz_test --data_name aitz_test

# gui_odyssey_test
python run_predict_minicpm.py --model_path /path/to/model --output_dir ./eval_results/gui_odyssey_test --data_name gui_odyssey_test

# chinese_app_test
python run_predict_minicpm.py --model_path /path/to/model --output_dir ./eval_results/chinese_app_test --data_name chinese_app_test

# android_control_high_test
python run_predict_minicpm.py --model_path /path/to/model --output_dir ./eval_results/android_control_high_test --data_name android_control_high_test

# android_control_low_test
python run_predict_minicpm.py --model_path /path/to/model --output_dir ./eval_results/android_control_low_test --data_name android_control_low_test

# Grounding
……
```

### Eval

```
# aitz_test
python run_eval_agent.py --input_path ./eval_results/aitz_test/all.jsonl --output_dir ./eval_results/aitz_test/results --data_name aitz_test

# gui_odyssey_test
python run_eval_agent.py --input_path ./eval_results/gui_odyssey_test/all.jsonl --output_dir ./eval_results/gui_odyssey_test/results --data_name gui_odyssey_test

# chinese_app_test
python run_eval_agent.py --input_path ./eval_results/chinese_app_test/all.jsonl --output_dir ./eval_results/chinese_app_test/results --data_name chinese_app_test

# android_control_high_test
python run_eval_agent.py --input_path ./eval_results/android_control_high_test/all.jsonl --output_dir ./eval_results/android_control_high_test/results --data_name android_control_test --android_control_high_test

# android_control_low_test
python run_eval_agent.py --input_path ./eval_results/android_control_low_test/all.jsonl --output_dir ./eval_results/android_control_low_test/results --data_name android_control_test_low --android_control_low_test

# Grounding
……
```

---

## Qwen2.5-VL-7B

### Inference

```
# aitz_test
……
```

### Eval

```
# aitz_test
……
```

---

## UI-TARS-7B-SFT

### Inference

```
# aitz_test
……
```

### Eval

```
# aitz_test
……
```

---

## OS-Genesis-7B-AC

### Inference

```
# aitz_test
……
```

### Eval

```
# aitz_test
……
```

---

## Aguvis-7B-720P

### Inference

```
# aitz_test
……
```

### Eval

```
# aitz_test
……
```

---

## OdysseyAgent-Random

### Inference

```
# aitz_test
……
```

### Eval

```
# aitz_test
……
```
