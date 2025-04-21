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
| MiniCPM-Agent | 94.39                  | 90.20                  | 77.70                   | 69.17                   | 90.85          | 74.96          | 85.71   | 76.38   | 96.86          | 91.28          |
|               |                        |                        |                         |                         |                |                |         |         |                |                |
|               |                        |                        |                         |                         |                |                |         |         |                |                |
|               |                        |                        |                         |                         |                |                |         |         |                |                |
|               |                        |                        |                         |                         |                |                |         |         |                |                |
|               |                        |                        |                         |                         |                |                |         |         |                |                |

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
