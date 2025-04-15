# Evaluation Scripts

## Grounding Benchmark

| Model         | fun2box | text2box | box2text |
| ------------- | ------- | -------- | -------- |
| MiniCPM-Agent |         |          |          |
|               |         |          |          |

## Agent Benchmark

| Dataset       | Android Control-Low TM | Android Control-Low EM | Android Control-High TM | Android Control-High EM | GUI-Odyssey TM | GUI-Odyssey EM | AITZ TM | AITZ EM | Chinese APP TM | Chinese APP EM |
| ------------- | ---------------------- | ---------------------- | ----------------------- | ----------------------- | -------------- | -------------- | ------- | ------- | -------------- | -------------- |
| MiniCPM-Agent |                        |                        |                         |                         |                |                |         |         |                |                |
|               |                        |                        |                         |                         |                |                |         |         |                |                |
|               |                        |                        |                         |                         |                |                |         |         |                |                |
|               |                        |                        |                         |                         |                |                |         |         |                |                |
|               |                        |                        |                         |                         |                |                |         |         |                |                |
|               |                        |                        |                         |                         |                |                |         |         |                |                |

## MiniCPM-Agent

### Inference

```
# aitz_test
python run_predict_minicpm.py --model_path /path/to/model --output_dir /path/to/save --data_name aitz_test

# gui_odyssey_test
python run_predict_minicpm.py --model_path /path/to/model --output_dir /path/to/save --data_name gui_odyssey_test

# domestic_test
python run_predict_minicpm.py --model_path /path/to/model --output_dir /path/to/save --data_name domestic_test

# android_control_test
python run_predict_minicpm.py --model_path /path/to/model --output_dir /path/to/save --data_name android_control_test

# android_control_test_low
python run_predict_minicpm.py --model_path /path/to/model --output_dir /path/to/save --data_name android_control_test_low

# Grounding
……
```

### Eval

```
# aitz_test
python run_eval_minicpm.py --input_path /path/to/inference_results --output_dir /path/to/save --data_name aitz_test

# gui_odyssey_test
python run_eval_minicpm.py --input_path /path/to/inference_results --output_dir /path/to/save --data_name gui_odyssey_test

# domestic_test
python run_eval_minicpm.py --input_path /path/to/inference_results --output_dir /path/to/save --data_name domestic_test

# android_control_test
python run_eval_minicpm.py --input_path /path/to/inference_results --output_dir /path/to/save --data_name android_control_test --eval_android_control

# android_control_test_low
python run_eval_minicpm.py --input_path /path/to/inference_results --output_dir /path/to/save --data_name android_control_test_low --eval_android_control

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
