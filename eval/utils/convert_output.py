import json
import os
import jsonschema
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


# Get the absolute path of the current file
current_file_path = os.path.abspath(__file__)
schema_dir = os.path.dirname(os.path.dirname(current_file_path))
EXTRACT_SCHEMA = json.load(open(os.path.join(schema_dir, 'utils/schema', 'schema_for_extraction.json'), encoding="utf-8"))


def load_json_data(file_path):
    data = []
    # Determine file type, support both JSON and JSONL
    if file_path.endswith('.json'):
        # Handle JSON file
        with open(file_path, 'r') as file:
            data = json.load(file)
    elif file_path.endswith('.jsonl'):
        # Handle JSONL file
        with open(file_path, 'r') as file:
            first_line = file.readline().strip()
            try:
                json.loads(first_line)
                data.append(json.loads(first_line))
            except json.JSONDecodeError:
                pass
            for line in file:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    return data


def parse_action(data):
    try:
        jsonschema.validate(data, EXTRACT_SCHEMA)
        
        actions = {}
        parameters = {}
        status = data.get("STATUS", "continue")  # Default value

        # Define actions
        action_keys = ["POINT", "to", "PRESS", "TYPE"]
        
        # Extract actions
        for key in action_keys:
            if key in data:
                actions[key] = data[key]
        
        # Extract global parameters
        parameters["duration"] = data.get("duration", EXTRACT_SCHEMA["properties"]["duration"]["default"])

        # Handle "to" parameter, if present
        if "to" in data:
            parameters["to"] = data["to"]
            
        return actions, parameters, status

    except Exception as e:
        print('Error, JSON is NOT valid according to the schema.')
        return None, None, None


# Use multiprocessing to speed up processing
def process_step(args):
    task, episode_id, step_id, pred, base_path = args
    try:
        actions, parameters, status = parse_action(pred)

        transformed_entry = {
            "action_predict": {
                "COA": {
                    "txt": {
                        "ACTION": actions,
                        "ARGS": parameters,
                        "STATUS": status
                    },
                }
            }
        }

        folder = f"{task}-{episode_id}"
        file_name = f"{folder}_{step_id}.json"
        output_file_path = os.path.join(base_path, folder, file_name)

        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            json.dump(transformed_entry, output_file, indent=4, ensure_ascii=False)
        
        return f"Saved transformed entry to: {output_file_path}"
    except Exception as e:
        return f"Error processing step {step_id} in episode {episode_id}: {e}"


# # Multi-threaded version
def convert2aitz(input_path, output_path, max_workers=None):
    data = load_json_data(input_path)
    base_path = os.path.join(output_path)
    folders = set()
    tasks = []
    for item in data:
        task = item.get("category", item.get("subset", "unknown"))
        episode_id = item.get("episode_id", "unknown")
        steps = item.get("steps", [item])

        for index, each_step in enumerate(steps):
            step_id = index if "steps" in item else each_step.get("step_id", index)
            folder = f"{task}-{episode_id}"
            folders.add(folder)
            pred = each_step.get("pred", {})
            tasks.append((task, episode_id, step_id, pred, base_path))
    
    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        os.makedirs(folder_path, exist_ok=True)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_step, task_args) for task_args in tasks]        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing steps"):
            result = future.result()
            print(result)


# # Single-threaded version
def convert2aitz_single_thread(input_path, output_path):
    data = load_json_data(input_path)    
    base_path = os.path.join(output_path)

    for item in data:
        task = item.get("category", "unknown")
        episode_id = item.get("episode_id", "unknown")
        steps = item.get("steps", [item])

        for index, each_step in enumerate(steps):
            step_id = index if "steps" in item else each_step.get("step_id", index)

            actions, parameters, status = parse_action(each_step["pred"])

            transformed_entry = {
                "action_predict": {
                    "COA": {
                        "txt": {
                            "ACTION": actions,
                            "ARGS": parameters,
                            "STATUS": status
                        },
                    }
                }
            }
            folder = f"{task}-{episode_id}"
            file_name = f"{folder}_{step_id}.json"
            folder_path = os.path.join(base_path, folder)
            output_path = os.path.join(folder_path, file_name)

            os.makedirs(folder_path, exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as output_file:
                json.dump(transformed_entry, output_file, indent=4, ensure_ascii=False)

            print(f"Saved transformed entry to: {output_path}")
