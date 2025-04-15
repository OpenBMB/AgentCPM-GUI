import json
from jsonschema import validate, ValidationError
import os

# Get the absolute path of the current file
current_file_path = os.path.abspath(__file__)
schema_dir = os.path.dirname(current_file_path) 
# This is the schema for extracting actions, not the schema for training LLM — the latter is a subset of the former.
schema = json.load(open(os.path.join(schema_dir, 'schema_for_extraction.json'), encoding="utf-8"))

# test cases
test_cases = [
    {
        "name": "Valid Case: Only POINT",
        "data": {
            "POINT": [500, 300],
            "duration": 300,
            "STATUS": "continue"
        },
        "expected": True
    },
    {
        "name": "Valid Case: POINT with to (direction)",
        "data": {
            "POINT": [500, 300],
            "to": "left",
            "duration": 300,
            "STATUS": "continue"
        },
        "expected": True
    },
    {
        "name": "Valid Case: POINT with to (Location)",
        "data": {
            "POINT": [500, 300],
            "to": [600, 400],
            "duration": 300,
            "STATUS": "start"
        },
        "expected": True
    },
    {
        "name": "Valid Case: Only PRESS",
        "data": {
            "PRESS": "HOME",
            "duration": 200,
            "STATUS": "finish"
        },
        "expected": True
    },
    {
        "name": "Valid Case: Only TYPE",
        "data": {
            "TYPE": "Hello, World!",
            "duration": 250,
            "STATUS": "satisfied"
        },
        "expected": True
    },
    {
        "name": "Invalid Case: to without POINT",
        "data": {
            "to": "up",
            "duration": 300,
            "STATUS": "continue"
        },
        "expected": False
    },
    {
        "name": "Invalid Case: POINT with PRESS",
        "data": {
            "POINT": [500, 300],
            "PRESS": "HOME",
            "duration": 300,
            "STATUS": "continue"
        },
        "expected": False
    },
    {
        "name": "Invalid Case: PRESS with TYPE",
        "data": {
            "PRESS": "BACK",
            "TYPE": "Some text",
            "duration": 200,
            "STATUS": "continue"
        },
        "expected": False
    },
    {
        "name": "Invalid Case: Multiple Actions",
        "data": {
            "POINT": [500, 300],
            "PRESS": "HOME",
            "TYPE": "Hello",
            "duration": 300,
            "STATUS": "continue"
        },
        "expected": False
    },
    {
        "name": "Invalid Case: POINT with invalid to value",
        "data": {
            "POINT": [500, 300],
            "to": "invalid_direction",
            "duration": 300,
            "STATUS": "continue"
        },
        "expected": False
    },
    {
        "name": "Invalid Case: PRESS with invalid enum",
        "data": {
            "PRESS": "INVALID_KEY",
            "duration": 200,
            "STATUS": "continue"
        },
        "expected": False
    },
    {
        "name": "Invalid Case: LOCATION with out of range coordinates",
        "data": {
            "POINT": [1500, 300],
            "duration": 300,
            "STATUS": "continue"
        },
        "expected": False
    },
    {
        "name": "Invalid Case: Missing STATUS (should default to continue)",
        "data": {
            "PRESS": "ENTER",
            "duration": 200
        },
        "expected": True  # STATUS has a default, so it's valid
    },
    {
        "name": "Invalid Case: Additional Property",
        "data": {
            "PRESS": "HOME",
            "extra_property": "not_allowed",
            "duration": 200,
            "STATUS": "continue"
        },
        "expected": False
    },
    {
        "name": "Valid Case: POINT with to and default STATUS",
        "data": {
            "POINT": [400, 200],
            "to": "down",
            "duration": 300
        },
        "expected": True
    },
        {
        "name": "Valid Case: POINT with default STATUS",
        "data": {
            "POINT": [400, 200],
        },
        "expected": True
    },
    {
        "name": "Invalid Case: Negative duration",
        "data": {
            "PRESS": "HOME",
            "duration": -100,
            "STATUS": "continue"
        },
        "expected": False
    },
    {
        "name": "Valid Case: Only STATUS 'finish'",
        "data": {
            "STATUS": "finish"
        },
        "expected": True
    },
    {
        "name": "Valid Case: Only STATUS 'satisfied'",
        "data": {
            "STATUS": "finish"
        },
        "expected": True
    },
    {
        "name": "Valid Case: Only STATUS 'impossible'",
        "data": {
            "STATUS": "finish"
        },
        "expected": True
    },
    {
        "name": "Valid Case: Only STATUS 'interrupt'",
        "data": {
            "STATUS": "finish"
        },
        "expected": True
    },
    {
        "name": "Valid Case: Only STATUS 'need_feedback'",
        "data": {
            "STATUS": "finish"
        },
        "expected": True
    },
    {
        "name": "Valid Case: POINT at boundary (0,0)",
        "data": {
            "POINT": [0, 0],
            "duration": 200,
            "STATUS": "continue"
        },
        "expected": True
    },
    {
        "name": "Valid Case: POINT at boundary (1000,1000)",
        "data": {
            "POINT": [1000, 1000],
            "duration": 200,
            "STATUS": "start"
        },
        "expected": True
    },
    {
        "name": "Valid Case: duration missing (should default to 200)",
        "data": {
            "PRESS": "BACK",
            "STATUS": "finish"
        },
        "expected": True
    },
    {
        "name": "Valid Case: to as Location with boundary coordinates",
        "data": {
            "POINT": [500, 500],
            "to": [0, 1000],
            "duration": 250,
            "STATUS": "satisfied"
        },
        "expected": True
    },
    {
        "name": "Invalid Case: duration as string",
        "data": {
            "PRESS": "ENTER",
            "duration": "300",
            "STATUS": "continue"
        },
        "expected": False
    },
    {
        "name": "Invalid Case: POINT with one coordinate missing",
        "data": {
            "POINT": [500],
            "duration": 300,
            "STATUS": "continue"
        },
        "expected": False
    },
    {
        "name": "Invalid Case: to as array with one element",
        "data": {
            "POINT": [500, 300],
            "to": [600],
            "duration": 300,
            "STATUS": "continue"
        },
        "expected": False
    },
    {
        "name": "Invalid Case: PRESS as null",
        "data": {
            "PRESS": None,
            "duration": 200,
            "STATUS": "continue"
        },
        "expected": False
    },
    {
        "name": "Invalid Case: TYPE as null",
        "data": {
            "TYPE": None,
            "duration": 200,
            "STATUS": "continue"
        },
        "expected": False
    },
    {
        "name": "Invalid Case: 'continue' STATUS missing other action",
        "data": {
            "STATUS": "continue"
        },
        "expected": False
    },
    {
        "name": "Invalid Case: to as empty array",
        "data": {
            "POINT": [500, 300],
            "to": [],
            "duration": 300,
            "STATUS": "continue"
        },
        "expected": False
    },
    {
        "name": "Invalid Case: POINT as non-array",
        "data": {
            "POINT": "500,300",
            "duration": 300,
            "STATUS": "continue"
        },
        "expected": False
    },
    {
        "name": "Valid Case: POINT and to with minimum duration (0)",
        "data": {
            "POINT": [500, 300],
            "to": "up",
            "duration": 0,
            "STATUS": "continue"
        },
        "expected": True
    },
    {
        "name": "Valid Case: POINT and to with maximum duration (10000)",
        "data": {
            "POINT": [500, 300],
            "to": "down",
            "duration": 10000,
            "STATUS": "finish"
        },
        "expected": True
    },
    {
        "name": "Invalid Case: POINT with non-integer values",
        "data": {
            "POINT": [500.5, "300"],
            "duration": 300,
            "STATUS": "continue"
        },
        "expected": False
    },
    {
        "name": "Invalid Case: to as direction and POINT missing",
        "data": {
            "to": "left",
            "duration": 300,
            "STATUS": "continue"
        },
        "expected": False
    },
    {
        "name": "Invalid Case: to as invalid Location array size (3 elements)",
        "data": {
            "POINT": [500, 300],
            "to": [600, 400, 200],
            "duration": 300,
            "STATUS": "continue"
        },
        "expected": False
    },
    {
        "name": "Valid Case: Only STATUS with duration",
        "data": {
            "STATUS": "finish",
            "duration": 500
        },
        "expected": True
    },
    {
        "name": "Invalid Case: STATUS with invalid enum value",
        "data": {
            "STATUS": "unknown_status",
            "duration": 200
        },
        "expected": False
    },
    {
        "name": "Valid Case: POINT with to as direction and missing duration (should default)",
        "data": {
            "POINT": [400, 200],
            "to": "right",
            "STATUS": "start"
        },
        "expected": True
    },
    {
        "name": "Valid Case: PRESS with missing duration (should default)",
        "data": {
            "PRESS": "APPSELECT",
            "STATUS": "need_feedback"
        },
        "expected": True
    },
    {
        "name": "Valid Case: TYPE with missing duration (should default)",
        "data": {
            "TYPE": "Sample Text",
            "STATUS": "impossible"
        },
        "expected": True
    },
    {
        "name": "Invalid Case: POINT as empty array",
        "data": {
            "POINT": [],
            "duration": 300,
            "STATUS": "continue"
        },
        "expected": False
    },
    {
        "name": "Invalid Case: POINT with negative coordinates",
        "data": {
            "POINT": [-100, 300],
            "duration": 300,
            "STATUS": "continue"
        },
        "expected": False
    },
    {
        "name": "Invalid Case: PRESS with lowercase value",
        "data": {
            "PRESS": "home",
            "duration": 200,
            "STATUS": "continue"
        },
        "expected": False
    },
    {
        "name": "Invalid Case: TYPE with empty string",
        "data": {
            "TYPE": "",
            "duration": 200,
            "STATUS": "continue"
        },
        "expected": True  # Empty string is valid as per schema
    },
    {
        "name": "Valid Case: POINT with to as Location and missing duration (should default)",
        "data": {
            "POINT": [500, 500],
            "to": [600, 600],
            "STATUS": "start"
        },
        "expected": True
    },
    {
        "name": "Valid Case: POINT at boundary (0,0)",
        "data": {
            "POINT": [0, 0],
            "duration": 200,
            "STATUS": "continue"
        },
        "expected": True
    },
    {
        "name": "Valid Case: POINT at boundary (1000,1000)",
        "data": {
            "POINT": [1000, 1000],
            "duration": 200,
            "STATUS": "start"
        },
        "expected": True
    },
    {
        "name": "Valid Case: duration missing (should default to 200)",
        "data": {
            "PRESS": "BACK",
            "STATUS": "finish"
        },
        "expected": True
    },
    {
        "name": "Valid Case: to as Location with boundary coordinates",
        "data": {
            "POINT": [500, 500],
            "to": [0, 1000],
            "duration": 250,
            "STATUS": "satisfied"
        },
        "expected": True
    },
    {
        "name": "Invalid Case: duration as string",
        "data": {
            "PRESS": "ENTER",
            "duration": "300",
            "STATUS": "continue"
        },
        "expected": False
    },
    {
        "name": "Invalid Case: POINT with one coordinate missing",
        "data": {
            "POINT": [500],
            "duration": 300,
            "STATUS": "continue"
        },
        "expected": False
    },
    {
        "name": "Invalid Case: to as array with one element",
        "data": {
            "POINT": [500, 300],
            "to": [600],
            "duration": 300,
            "STATUS": "continue"
        },
        "expected": False
    },
    {
        "name": "Invalid Case: PRESS as null",
        "data": {
            "PRESS": None,
            "duration": 200,
            "STATUS": "continue"
        },
        "expected": False
    },
    {
        "name": "Invalid Case: TYPE as null",
        "data": {
            "TYPE": None,
            "duration": 200,
            "STATUS": "continue"
        },
        "expected": False
    },
    {
        "name": "Invalid Case: 'continue' STATUS missing other action",
        "data": {
            "STATUS": "continue"
        },
        "expected": False
    },
    {
        "name": "Invalid Case: 'start' STATUS missing other action",
        "data": {
            "STATUS": "start"
        },
        "expected": False
    },
    {
        "name": "Invalid Case: Empty object",
        "data": {},
        "expected": False
    },
    {
        "name": "Invalid Case: to as empty array",
        "data": {
            "POINT": [500, 300],
            "to": [],
            "duration": 300,
            "STATUS": "continue"
        },
        "expected": False
    },
    {
        "name": "Invalid Case: POINT as non-array",
        "data": {
            "POINT": "500,300",
            "duration": 300,
            "STATUS": "continue"
        },
        "expected": False
    },
    {
        "name": "Valid Case: POINT and to with minimum duration (0)",
        "data": {
            "POINT": [500, 300],
            "to": "up",
            "duration": 0,
            "STATUS": "continue"
        },
        "expected": True
    },
    {
        "name": "Valid Case: POINT and to with maximum duration (10000)",
        "data": {
            "POINT": [500, 300],
            "to": "down",
            "duration": 10000,
            "STATUS": "finish"
        },
        "expected": True
    },
    {
        "name": "Invalid Case: POINT with non-integer values",
        "data": {
            "POINT": [500.5, "300"],
            "duration": 300,
            "STATUS": "continue"
        },
        "expected": False
    },
    {
        "name": "Invalid Case: to as direction and POINT missing",
        "data": {
            "to": "left",
            "duration": 300,
            "STATUS": "continue"
        },
        "expected": False
    },
    {
        "name": "Invalid Case: to as invalid Location array size (3 elements)",
        "data": {
            "POINT": [500, 300],
            "to": [600, 400, 200],
            "duration": 300,
            "STATUS": "continue"
        },
        "expected": False
    },
    {
        "name": "Valid Case: Only STATUS with duration",
        "data": {
            "STATUS": "finish",
            "duration": 500
        },
        "expected": True
    },
    {
        "name": "Invalid Case: STATUS with invalid enum value",
        "data": {
            "STATUS": "unknown_status",
            "duration": 200
        },
        "expected": False
    },
    {
        "name": "Valid Case: POINT with to as direction and missing duration (should default)",
        "data": {
            "POINT": [400, 200],
            "to": "right",
            "STATUS": "start"
        },
        "expected": True
    },
    {
        "name": "Valid Case: PRESS with missing duration (should default)",
        "data": {
            "PRESS": "APPSELECT",
            "STATUS": "need_feedback"
        },
        "expected": True
    },
    {
        "name": "Valid Case: TYPE with missing duration (should default)",
        "data": {
            "TYPE": "Sample Text",
            "STATUS": "impossible"
        },
        "expected": True
    },
    {
        "name": "Invalid Case: POINT as empty array",
        "data": {
            "POINT": [],
            "duration": 300,
            "STATUS": "continue"
        },
        "expected": False
    },
    {
        "name": "Invalid Case: POINT with negative coordinates",
        "data": {
            "POINT": [-100, 300],
            "duration": 300,
            "STATUS": "continue"
        },
        "expected": False
    },
    {
        "name": "Invalid Case: PRESS with lowercase value",
        "data": {
            "PRESS": "home",
            "duration": 200,
            "STATUS": "continue"
        },
        "expected": False
    },
    {
        "name": "Invalid Case: TYPE with empty string",
        "data": {
            "TYPE": "",
            "duration": 200,
            "STATUS": "continue"
        },
        "expected": True  # Empty string is valid as per schema
    },
    {
        "name": "Valid Case: POINT with to as Location and missing duration (should default)",
        "data": {
            "POINT": [500, 500],
            "to": [600, 600],
            "STATUS": "start"
        },
        "expected": True
    },
    {
        "name": "Valid Case: only duration (just wait)",
        "data": {
            "duration": 200,
        },
        "expected": True
    },
    {
        "name": "Valid Case: POINT with to as Location and missing duration (should default), with thought",
        "data": {
            "POINT": [500, 500],
            "to": [600, 600],
            "STATUS": "start",
            "thought": "I am thinking"
        },
        "expected": True
    },

]

def run_tests(schema, test_cases):
    print("Starting test cases...\n")
    for idx, test in enumerate(test_cases, 1):
        data = test["data"]
        expected = test["expected"]
        name = test["name"]
        try:
            validate(instance=data, schema=schema)
            result = True
            error_message = ""
        except ValidationError as e:
            result = False
            error_message = e.message

        status = "PASS" if result == expected else "FAIL"
        print(f"Test Case {idx}: {name}")
        print(f"  Expected Result: {'Valid' if expected else 'Invalid'}")
        print(f"  Actual Result: {'Valid' if result else 'Invalid'}")
        if status == "FAIL":
            print(f"  Error Message: {error_message}")
        print(f"  Test Status: {status}\n")

    print("Testing completed.")


if __name__ == "__main__":
    run_tests(schema, test_cases)