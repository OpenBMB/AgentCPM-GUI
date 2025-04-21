from colorama import init, Fore, Style
import os
from utils.action_type import ActionType


def annotate_and_save_image(img_path, output_folder, gt_action_type, gt_action_detail, pd_action_type, pd_action_detail, type_match, exact_match, subset, episode_id, step_id, task_desc):
    """Save an annotated image with action details to the specified folder."""
    # Load the image and get its dimensions
    image = Image.open(img_path)
    draw = ImageDraw.Draw(image)

    # Dynamically compute font size based on image height
    base_height = 1080  # Reference height, e.g., 1080p
    font_size = max(12, int(image.height / base_height * 20))  # Ensure minimum font size is 12

    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    try:
        font = ImageFont.truetype(os.path.join(current_dir, './SimHei.ttf'), font_size)
    except IOError:
        # If the specified font file does not exist, use the default font
        font = ImageFont.load_default()

    w, h = image.width, image.height

    # Create annotation text
    annotation_text = (
        f"taskDesc: {task_desc}\n"
        f"taskID: {subset}{episode_id}_{step_id}\n"
        f"GT action: {gt_action_type}\n"
        f"GT detail: {gt_action_detail}\n"
        f"PD action: {pd_action_type}\n"
        f"PD detail: {pd_action_detail}\n"
        f"type_match: {'Yes' if type_match else 'No'}\n"
        f"exac_match: {'Yes' if exact_match else 'No'}"
    )

    # Calculate text size and wrap lines if necessary
    max_width = w - 20  # Max width for the text
    lines = []
    for line in annotation_text.split('\n'):
        # Split line by words to check width
        words = line.split()
        current_line = ""
        for word in words:
            # Check width after adding a word
            test_line = current_line + " " + word if current_line else word
            if draw.textlength(test_line, font=font) > max_width:
                # If line is too long, start a new line
                lines.append(current_line)
                current_line = word
            else:
                current_line = test_line
        lines.append(current_line)  # Add the final line

    # Draw each line on the image
    y_text = 10
    line_spacing = int(font_size * 1.2)  # Line spacing is 1.2 times the font size
    for line in lines:
        draw.text((10, y_text), line, font=font, fill='red')
        y_text += line_spacing  # Move to next line position

    # Draw rectangle and point based on conditions
    if pd_action_type == 'click' and type_match:
        if isinstance(gt_action_detail, (list, tuple)) and len(gt_action_detail) == 4:
            ymin, xmin, height, width = gt_action_detail  # Parse GT action details
            pd_x = pd_action_detail.get("x", 0) * w
            pd_y = pd_action_detail.get("y", 0) * h
            gt_box = [xmin * w, ymin * h, (xmin + width) * w, (ymin + height) * h]
            draw.rectangle(gt_box, outline="red", width=max(1, int(font_size / 5)))  # Adjust line width dynamically
            point_radius = max(5, int(font_size / 2))  # Adjust point radius dynamically
            draw.ellipse(
                (pd_x - point_radius, pd_y - point_radius, pd_x + point_radius, pd_y + point_radius),
                fill="red",
                outline="blue",
                width=max(1, int(font_size / 10))
            )

    # Save the annotated image to the output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    output_file_name = os.path.basename(img_path).replace('.png', '_annotated.png')
    output_path = os.path.join(output_folder, output_file_name)
    image.save(output_path)

    return output_path


def get_dataset_dir(data_name):
    data_list = ['aitz_test', 'chinese_app_test', 'gui_odyssey_test', 'android_control_high_test', 'android_control_low_test']
    assert data_name in data_list, "Error, unkonw eval dataset."
    data_split = None
    data_dir = None
    data_subset = None

    current_file_path = os.path.abspath(__file__)
    data_dir = os.path.dirname(os.path.dirname(current_file_path))
    
    match data_name:
        case 'aitz_test':
            data_dir = os.path.join(data_dir, "eval_data", "aitz_test")
            data_split = "test"
            data_subset = ["general", "install", "web_shopping", "google_apps"]
        case 'chinese_app_test':
            data_dir = os.path.join(data_dir, "eval_data", "chinese_app_test")
            data_split = "test"
            data_subset = ["domestic"]
        case 'gui_odyssey_test':
            data_dir = os.path.join(data_dir, "eval_data", "odyssey")
            data_split = "test"
            data_subset = ["odyssey"]
        case 'android_control_high_test':
            data_dir = os.path.join(data_dir, "eval_data", "android_control_high_test")
            data_split = "test"
            data_subset = ["android_control"]
        case 'android_control_low_test':
            data_dir = os.path.join(data_dir, "eval_data", "android_control_low_test")
            data_split = "test"
            data_subset = ["android_control"]

    return data_dir, data_split, data_subset
