from util.utils import get_som_labeled_img, check_ocr_box, get_caption_model_processor, get_yolo_model
import torch
from ultralytics import YOLO
from PIL import Image
import os
import base64
import matplotlib.pyplot as plt
import io
import json

device = 'cuda'
model_path='weights/icon_detect/model.pt'

som_model = get_yolo_model(model_path)

som_model.to(device)
print('model to {}'.format(device))

caption_model_processor = get_caption_model_processor(model_name="florence2", model_name_or_path="weights/icon_caption_florence", device=device)
source_dir = "your/path/to/original/image"
target_dir = "your/path/to/annotated/image"


for image_dir in os.listdir(source_dir):
    if image_dir.endswith(".jpeg"):
        image_path = os.path.join(source_dir, image_dir)
        output_dir = image_path.replace(source_dir, target_dir)
        if os.path.exists(output_dir):
            continue
        else:
            image = Image.open(image_path)
            image_rgb = image.convert('RGB')
            print('image size:', image.size)

            box_overlay_ratio = max(image.size) / 3200
            draw_bbox_config = {
                'text_scale': 0.8 * box_overlay_ratio,
                'text_thickness': max(int(2 * box_overlay_ratio), 1),
                'text_padding': max(int(3 * box_overlay_ratio), 1),
                'thickness': max(int(3 * box_overlay_ratio), 1),
            }
            BOX_TRESHOLD = 0.05

            try:
                ocr_bbox_rslt, is_goal_filtered = check_ocr_box(image_path, display_img = False, output_bb_format='xyxy', goal_filtering=None, easyocr_args={'paragraph': False, 'text_threshold':0.9}, use_paddleocr=True)
                text, ocr_bbox = ocr_bbox_rslt
                dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(image_path, som_model, BOX_TRESHOLD = BOX_TRESHOLD, output_coord_in_ratio=True, ocr_bbox=ocr_bbox,draw_bbox_config=draw_bbox_config, caption_model_processor=caption_model_processor, ocr_text=text,use_local_semantics=True, iou_threshold=0.7, scale_img=False, batch_size=128)

                image = Image.open(io.BytesIO(base64.b64decode(dino_labled_img)))
                if not os.path.exists(output_dir):
                    mkdir = '/'.join(output_dir.split('/')[:-1])
                    os.makedirs(mkdir, exist_ok=True)
                image.save(output_dir)
                content_output_dir = output_dir.replace('.jpeg', '.json')
                if not os.path.exists(content_output_dir):
                    mkdir = '/'.join(content_output_dir.split('/')[:-1])
                    os.makedirs(mkdir, exist_ok=True)
                with open(content_output_dir, "w", encoding="utf-8") as f:
                    json.dump(parsed_content_list,f,indent=2)
            except:
                print(f"未成功处理：{image_path}")