import os
import json
import numpy as np
import Levenshtein
import math
from PIL import Image, ImageDraw, ImageFont
from utils.action_type import ActionType
from utils.utils import annotate_and_save_image
from typing import List, Union

 # Based on evaluator of Qwen 2.5 VL
 # https://github.com/QwenLM/Qwen2.5-VL/issues/904
 # https://gist.github.com/LukeForeverYoung/274a073ca77c9dc46022cb8cc5382223
 # https://gist.github.com/LukeForeverYoung/1f5d19495788de0d905c5ac6341153f5

# Get the absolute path of the current file
current_file_path = os.path.abspath(__file__)
schema_dir = os.path.dirname(os.path.dirname(current_file_path))
EXTRACT_SCHEMA = json.load(open(os.path.join(schema_dir, 'utils/schema', 'schema_for_extraction.json'), encoding="utf-8"))

# CONSTANTS
_TAP_DISTANCE_THRESHOLD = 0.14  # Fraction of the screen
_TAP_DISTANCE_THRESHOLD_AC = 0.04  # for android control, align with qwen's code.
_SWIPE_DISTANCE_THRESHOLD = 0.04 # Interval determining if an action is a tap or a swipe.
ANNOTATION_WIDTH_AUGMENT_FRACTION= 1.2 # aitw set it to 1.4, aitz and qwen 2.5 vl set it to 1.2.
ANNOTATION_HEIGHT_AUGMENT_FRACTION= 1.2 # We follow qwen setting.
default_duration = EXTRACT_SCHEMA["properties"]["duration"]["default"] # default 200


def _resize_annotation_bounding_boxes(
    annotation_position: Union[List[float], List[List[float]]],
    width_factor: float = 1.2,
    height_factor: float = 1.2,
):
    """Uniformly enlarge bbox(es) by the given factors."""

    def _resize(box: List[float]):
        y, x, h, w = box
        h_delta = (height_factor - 1) * h
        w_delta = (width_factor - 1) * w
        y = max(0, y - h_delta / 2)
        x = max(0, x - w_delta / 2)
        h = min(1, h + h_delta)
        w = min(1, w + w_delta)
        return [y, x, h, w]

    if not annotation_position:
        return []
    if isinstance(annotation_position[0], list):
        return [_resize(b) for b in annotation_position]
    return _resize(annotation_position)


def is_tap_action(normalized_start_yx, normalized_end_yx):
  distance = np.linalg.norm(np.array(normalized_start_yx) - np.array(normalized_end_yx))
  return distance <= _SWIPE_DISTANCE_THRESHOLD


def check_inside(x, y, bbox_list):
    bbox_array = np.array(bbox_list)
    y_min, x_min, height, width = bbox_array[:, 0], bbox_array[:, 1], bbox_array[:, 2], bbox_array[:, 3]
    y_max, x_max = y_min + height, x_min + width

    # Check whether (x, y) is inside any of the bounding boxes
    within_x = (x_min <= x) & (x <= x_max)
    within_y = (y_min <= y) & (y <= y_max)
    within_bbox = within_x & within_y

    if np.any(within_bbox):
        within_bbox_coords = bbox_array[within_bbox]
        return True, within_bbox_coords
    else:
        return False, None


def obtain_gt_bbox(coordinate, bbox_list, eval_android_control=False):
    x, y = coordinate['x'], coordinate['y']
    if len(bbox_list) == 0:
        return []

    if not eval_android_control:
        is_inside, bbox_inside = check_inside(x, y, bbox_list)
        if is_inside:
            return bbox_inside.tolist()
        else:
            return []
    else:
        def get_center_distance(box):
            ymin, xmin, h, w = box
            center_y = ymin + h/2
            center_x = xmin + w/2
            return ((center_y - y) ** 2 + (center_x - x) ** 2) ** 0.5

        sorted_boxes = sorted(bbox_list, key=get_center_distance)
        # return the 5 nearest bboxes
        return sorted_boxes[:5]


def _get_direction(point1, point2):
    try:
        x1, y1 = point1["x"], point1["y"]
        x2, y2 = point2["x"], point2["y"]

        assert x1 is not None
        assert x2 is not None
        assert y1 is not None
        assert y2 is not None

        vector = (x2 - x1, y2 - y1)
        vx, vy = vector
    except Exception as e:
        return "no direction"

    directions = {
        "up": (0, -1),
        "down": (0, 1),
        "left": (-1, 0),
        "right": (1, 0)
    }

    vector_length = math.sqrt(vx ** 2 + vy ** 2)
    if vector_length == 0:
        return "no direction"
    unit_vector = (vx / vector_length, vy / vector_length)

    max_cosine = -float('inf')
    closest_direction = None
    for direction, dir_vector in directions.items():
        dx, dy = dir_vector
        dir_length = math.sqrt(dx ** 2 + dy ** 2)
        cos_theta = (unit_vector[0] * dx + unit_vector[1] * dy) / dir_length
        if cos_theta > max_cosine:
            max_cosine = cos_theta
            closest_direction = direction

    return closest_direction

def get_direction(point, to):
    if isinstance(to, str):
        if to in ["up", "down", "left", "right"]:
            return to
        else:
            return "no direction"
    elif isinstance(to, list):
        try:
            point1 = {"x": point[0], "y": point[1]}
            point2 = {"x": to[0], "y": to[1]}
            return _get_direction(point1, point2)
        except Exception as e:
            return "no direction"

class ActionEvaluator(object):

    def __init__(self, save_dir, eval_android_control=False) -> None:
        self.save_dir = save_dir
        # compatible with aitz evaluator
        self.demo_mode = "COA"
        self.screen_mode = "txt"

        self._aitz_action_type_ = ActionType
        self._stop_status = [
          "finish",
          "satisfied",
          "impossible",
          "interrupt",
          "need_feedback"
        ]
        self.eval_android_control = eval_android_control

    def action_map(self, action_api: dict):
        action = action_api.get('ACTION', None)
        args = action_api.get('ARGS', None)
        status = action_api.get('STATUS', None)
        duration = args.get('duration', default_duration) if args else None

        if action is None and args is None and status is None:
            print('Schema error.')
            return None, {}
        elif status in self._stop_status:
            return "stop", {}
        elif "TYPE" in action:
            return "type", action['TYPE']
        elif "POINT" in action and "to" not in args and duration == default_duration: # click
            return "click", action['POINT']
        elif "POINT" in action and "to" in args and duration == default_duration: # swipte
            return "scroll", {"start": action['POINT'], "end": args['to']}
        elif "POINT" in action and "duration" in args and duration > default_duration: # long press
            return "long_point", {"coordinate": action['POINT'], "duration": args['duration']}
        elif "PRESS" in action:
            return "press", action['PRESS']
        elif "duration" in args: # pause and wait
            return "stop", args['duration']
        else:
            raise ValueError("Unknown action type.")

    def _parse_action_(self, pred, image_width=None, image_height=None):
        pd_action_type, pd_action_yx, pd_action_idx, pd_action_direction, pd_action_text, pd_action_button, pd_duration = (None, ) * 7

        pr = pred.get('action_predict', {})
        if self.demo_mode not in pr: return (None, ) * 7

        action = pr[self.demo_mode].get(self.screen_mode, {})
        if not action: return (None, ) * 7

        pd_action_type, pd_action_args = self.action_map(action)
        if pd_action_type is None: print('Unknown action: ', action)

        # scale factors
        scale_x = 1000
        scale_y = 1000

        if pd_action_type == "click":
            try:
                pd_action_yx = {"x": pd_action_args[0] / scale_x, "y": pd_action_args[1] / scale_y}
            except Exception as e:
                pd_action_yx = {"x": 0.0, "y": 0.0}
        elif pd_action_type == "long_point":
            try:
                pd_action_yx = {"x": pd_action_args["coordinate"][0] / scale_x, "y": pd_action_args["coordinate"][1] / scale_y}
            except Exception as e:
                pd_action_yx = {"x": 0.0, "y": 0.0}
        else:
            pd_action_yx = None

        # Not supporting click by id
        pd_action_idx = None

        # Process swipe
        pd_action_direction = get_direction(pd_action_args["start"], pd_action_args["end"]) if pd_action_type == "scroll" else None

        # Process text input
        pd_action_text = pd_action_args if pd_action_type == "type" else None

        # Process button press
        pd_action_button = pd_action_args.lower() if pd_action_type == "press" else None

        # Process long press
        pd_duration = pd_action_args["duration"] if pd_action_type == "long_point" else None

        # Treat pause and wait as normal stop

        return pd_action_type, pd_action_yx, pd_action_idx, pd_action_text, pd_action_button, pd_action_direction, pd_duration


    def _parse_answer_(self, gt):
        gt_cand_nodes=None
        gt_action_text=None
        gt_action_type=None
        gt_action_yx=None
        gt_action_direction=None
        gt_action_button=None
        gt_duration=None
        if gt['result_action_type'] == self._aitz_action_type_.TYPE:
            gt_action_type = "type"
            gt_action_text = gt['result_action_text']
        elif gt['result_action_type'] == self._aitz_action_type_.DUAL_POINT:  # Might be swipe or click
            normalized_start_yx = gt['result_touch_yx']
            normalized_start_yx = json.loads(normalized_start_yx)
            normalized_end_yx = gt['result_lift_yx']
            normalized_end_yx =json.loads(normalized_end_yx)

            if is_tap_action(normalized_start_yx, normalized_end_yx):
                gt_cand_nodes = json.loads(gt['ui_positions'])
                gt_action_type = "click"
                gt_action_yx = {"y": normalized_start_yx[0], "x": normalized_start_yx[1]}
            else:
                point1 = {"y": normalized_start_yx[0], "x": normalized_start_yx[1]}
                point2 = {"y": normalized_end_yx[0], "x": normalized_end_yx[1]}
                gt_action_type = "scroll"
                gt_action_direction = _get_direction(point1, point2)
        elif gt['result_action_type'] == self._aitz_action_type_.LONG_POINT:
            normalized_start_yx = gt['result_touch_yx']
            normalized_start_yx = json.loads(normalized_start_yx)
            normalized_end_yx = gt['result_lift_yx']
            normalized_end_yx =json.loads(normalized_end_yx)

            gt_cand_nodes = json.loads(gt['ui_positions'])
            gt_action_type = "long_point"
            gt_action_yx = {"y": normalized_start_yx[0], "x": normalized_start_yx[1]}
            gt_duration = gt['duration']
        elif gt['result_action_type'] == self._aitz_action_type_.PRESS_BACK:
            gt_action_type = "press"
            gt_action_button = "back"
        elif gt['result_action_type'] == self._aitz_action_type_.PRESS_HOME:
            gt_action_type = "press"
            gt_action_button = "home"
        elif gt['result_action_type'] == self._aitz_action_type_.PRESS_ENTER:
            gt_action_type = "press"
            gt_action_button = "enter"
        elif gt['result_action_type'] == self._aitz_action_type_.STATUS_TASK_COMPLETE or gt['result_action_type'] == self._aitz_action_type_.STATUS_TASK_IMPOSSIBLE:
            gt_action_type = "stop"
            gt_action_text = gt['result_action_text']
        elif gt['result_action_type'] == self._aitz_action_type_.NO_ACTION:
            gt_action_type = "stop"
            gt_duration = gt['duration']
        else:
            raise ValueError("Unknow action type.")

        return gt_action_type, gt_action_yx, gt_cand_nodes, \
               gt_action_text, gt_action_button, gt_action_direction, gt_duration

    def __call__(self, gt, pred, annotate_image=False):
        """ eval_single_step """
        pd_action_detail = None
        pixel_distance = None

        image_width, image_height = gt['image_width'], gt['image_height']

        subset, episode_id, step_id, task_desc = gt['subset'], gt['episode_id'], gt['step_id'], gt['instruction']

        # get ground truth information
        gt_action_type, gt_action_yx, gt_cand_nodes, \
            gt_action_text, gt_action_button, gt_action_direction, gt_duration = self._parse_answer_(gt)
        if not gt_action_type: print(gt['result_action_type'])
        gt_action_detail = {
            "click": gt_action_yx,
            "scroll": gt_action_direction,
            "type": gt_action_text,
            "press": gt_action_button,
            "long_point": gt_action_yx,
            "stop": "stop"
        }.get(gt_action_type, None)

        # get predict action information
        pd_action_type, pd_action_yx, pd_action_idx, \
            pd_action_text, pd_action_button, pd_action_direction, pd_duration = self._parse_action_(pred, image_width, image_height)
        pd_action_detail={
            "click": pd_action_yx,
            "scroll": pd_action_direction,
            "type": pd_action_text,
            "press": pd_action_button,
            "long_point": pd_action_yx,
            "stop": "stop"
        }.get(pd_action_type, None)

        # compute metrics
        hit_format = True if pd_action_type is not None else False # invalid actions are set to None when converting format
        type_match = (pd_action_type is not None and gt_action_type == pd_action_type)
        exact_match = False
        text_dist = None

        if type_match and (pd_action_type == "click" or pd_action_type == "long_point"):
            gt_cand_nodes = _resize_annotation_bounding_boxes(gt_cand_nodes, ANNOTATION_WIDTH_AUGMENT_FRACTION, ANNOTATION_HEIGHT_AUGMENT_FRACTION)
            gt_bbox = obtain_gt_bbox(gt_action_yx, gt_cand_nodes, self.eval_android_control)

            if gt_bbox == []:
                y_gt, x_gt = gt_action_yx["y"], gt_action_yx["x"]
                y_pd, x_pd = pd_action_yx["y"], pd_action_yx["x"]
                distance = np.linalg.norm(np.array([x_gt, y_gt]) - np.array([x_pd, y_pd]))
                exact_match = bool(distance <= (_TAP_DISTANCE_THRESHOLD_AC if self.eval_android_control else _TAP_DISTANCE_THRESHOLD))
                reference_point = gt_action_yx["x"], gt_action_yx["y"]
            else:
                reference_point = gt_action_yx["x"], gt_action_yx["y"]
                for bbox in gt_bbox:
                    ymin, xmin, height, width = bbox
                    ymax, xmax = ymin + height, xmin + width
                    exact_match = ((ymin <= pd_action_yx["y"] <= ymax) and (xmin <= pd_action_yx["x"] <= xmax))
                    if exact_match:
                        reference_point = (xmax + xmin) / 2, (ymax + ymin) / 2
                        break
                if not exact_match:
                    y_gt, x_gt = gt_action_yx["y"], gt_action_yx["x"]
                    y_pd, x_pd = pd_action_yx["y"], pd_action_yx["x"]
                    distance = np.linalg.norm(np.array([x_gt, y_gt]) - np.array([x_pd, y_pd]))
                    exact_match = bool(distance <= (_TAP_DISTANCE_THRESHOLD_AC if self.eval_android_control else _TAP_DISTANCE_THRESHOLD))
            # Calculate pixel mse, here the distance is calculated in the normalized space [0, 1000]
            pixel_distance = np.linalg.norm(np.array([pd_action_yx["x"], pd_action_yx["y"]])*1000 - np.array(reference_point)*1000)
        if type_match and pd_action_type == "scroll":
            exact_match = (pd_action_direction == gt_action_direction)

        if type_match and pd_action_type == "type":
            pd_text_norm = pd_action_text.lower().strip()
            gt_text_norm = gt_action_text.lower().strip()

            text_dist = Levenshtein.ratio(pd_text_norm, gt_text_norm)

            # align with Qwen‑2.5‑VL eval
            exact_match = (pd_text_norm in gt_text_norm or \
                        gt_text_norm in pd_text_norm)

        if type_match and pd_action_type == "press":
            exact_match = (pd_action_button == gt_action_button)

        if type_match and pd_action_type == "stop":
            exact_match = True

        # for visualization
        if annotate_image:
            output_folder = os.path.join(self.save_dir, "image_output")
            annotate_and_save_image(gt['image_full_path'], output_folder,
                                    gt_action_type, gt_action_detail,
                                    pd_action_type, pd_action_detail, type_match,
                                    exact_match,subset, episode_id, step_id, task_desc)

        if not type_match or (type_match and not exact_match):
            match_type = "No Type Match" if not type_match else "No Exact Match"
            print(f"\n{match_type}, pd action: {pd_action_type}, detail: {pd_action_detail}; gt action: {gt_action_type}, detail: {gt_action_detail}, {subset}_{episode_id}_{step_id}")

        return {
            "subset": subset,
            "episode_id": episode_id,
            "step_id": step_id,
            "answer": {
                "action_type": gt_action_type,
                "action_detail": gt_action_detail
            },
            "pred": {
                "action_type": pd_action_type,
                "action_detail": pd_action_detail
            },
            "type_match": type_match,
            "exact_match": exact_match,
            "text_dist": text_dist,
            "format_hit": hit_format,
            "pixel_distance": pixel_distance,
        }

    @staticmethod
    def compute_episode_metrics(episode_results):
        success, progress = [], []
        total_exact_matches = 0
        total_steps = 0
        for __, eplist in episode_results.items():
            ep_success, ep_progress = True, 0
            for ex in eplist:
                if ex['exact_match'] is True:
                    ep_progress += 1
                    total_exact_matches += 1
                else:
                    ep_success = False
                if not ep_success:
                    break
            success.append(ep_success)
            progress.append(ep_progress/len(eplist)*1.0)

        total_steps = 0
        for __, eplist in episode_results.items():
            for ex in eplist:
                total_steps += 1

        num_episodes = len(success)
        num_successes = sum(success)

        return {
            "total_episodes": num_episodes,
            "total_steps": total_steps,
            "num_successes": num_successes,
            "total_exact_matches": total_exact_matches,
            "success_rate": round(sum(success) / len(success), 4),
            "goal_progress": round(sum(progress) / len(progress), 4)}

    @staticmethod
    def compute_atomic_metrics(step_results):
        recorder = {
            'total':  {'count':0, 'type_match':0, 'exact_match':0, "hit": 0},
            # -------------------------------------------
            'CLICK':  {'count':0, 'type_match':0, 'exact_match':0},
            'TYPE':   {'count':0, 'type_match':0, 'exact_match':0, 'text_dist': []},
            'SCROLL': {'count':0, 'type_match':0, 'exact_match':0},
            'PRESS':  {'count':0, 'type_match':0, 'exact_match':0},
            'STOP':   {'count':0, 'type_match':0, 'exact_match':0},
            'LONG_POINT':   {'count':0, 'type_match':0, 'exact_match':0},
        }
        for step in step_results:
            recorder['total']['count'] += 1
            recorder['total']['hit'] += step.get('format_hit', 0)

            # Get action_type and ensure it is a string
            action_type = step.get('answer', {}).get('action_type')
            if isinstance(action_type, str):
                action_type = action_type.upper()
            else:
                action_type = ''

            if action_type in recorder:
                recorder[action_type]['count'] += 1
                recorder[action_type]['type_match'] += step.get('type_match', 0)
                recorder['total']['type_match'] += step.get('type_match', 0)
                recorder[action_type]['exact_match'] += step.get('exact_match', 0)
                recorder['total']['exact_match'] += step.get('exact_match', 0)
                if 'text_dist' in recorder[action_type] and step.get('text_dist') is not None:
                    recorder[action_type]['text_dist'].append(step['text_dist'])

        # Initialize scores dictionary, including counts and ratios
        scores = {
            metric_key: {
                'count': recorder[metric_key]['count'],
                'type_acc': round(
                    recorder[metric_key]['type_match'] / recorder[metric_key]['count'],
                    4
                ) if recorder[metric_key]['count'] > 0 else 0,
                'exact_acc': round(
                    recorder[metric_key]['exact_match'] / recorder[metric_key]['count'],
                    4
                ) if recorder[metric_key]['count'] > 0 else 0
            }
            for metric_key in ['total', 'CLICK', 'LONG_POINT', 'SCROLL', 'PRESS', 'STOP', 'TYPE']
        }

        # Calculate hit_rate
        scores['total']['hit_rate'] = round(
            recorder['total']['hit'] / recorder['total']['count'], 4
        ) if recorder['total']['count'] > 0 else 0

        # Calculate average text_dist for TYPE
        if recorder['TYPE']['text_dist']:
            scores['TYPE']['text_dist_avg'] = round(
                sum(recorder['TYPE']['text_dist']) / len(recorder['TYPE']['text_dist']), 4
            )
        else:
            scores['TYPE']['text_dist_avg'] = 0

        # Calculate pixel distance
        pixel_distances = [
            step['pixel_distance'] for step in step_results
            if step.get('pixel_distance') is not None
        ]

        median_pixel_distance = round(
            float(np.median(pixel_distances)), 4
        ) if pixel_distances else -1

        mean_pixel_distance = -1

        if pixel_distances:
            pixel_distances = np.array(pixel_distances)
            filtered_distances = pixel_distances[pixel_distances < 1e15]
            if len(filtered_distances) > 0:
                mean_pixel_distance = round(
                    float(np.mean(filtered_distances)), 4
                )

        scores['mean_pixel_distance'] = mean_pixel_distance
        scores['median_pixel_distance'] = median_pixel_distance

        return scores