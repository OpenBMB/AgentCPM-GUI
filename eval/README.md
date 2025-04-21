# Evaluation Scripts

## Grounding Benchmark

| Model         | fun2box | text2box | box2text |
| ------------- | ------- | -------- | -------- |
| MiniCPM-Agent |         |          |          |
|               |         |          |          |

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
