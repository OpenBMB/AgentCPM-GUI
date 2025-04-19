import argparse
import itertools
import json
import os
import random
import re
import time
from functools import partial

import torch
from internvl.model import load_model_and_tokenizer
from internvl.train.dataset import build_transform, dynamic_preprocess
from PIL import Image
from torchvision.ops.boxes import box_area
from tqdm import tqdm


def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def collate_fn(batches, tokenizer):
    pixel_values = torch.cat([_['pixel_values'] for _ in batches], dim=0)
    questions = [_['question'] for _ in batches]
    targets = [_['target'] for _ in batches]
    hws = [_['hw'] for _ in batches]
    return pixel_values, questions, targets, hws


class ChineseDataset(torch.utils.data.Dataset):

    def __init__(self, test, prompt, input_size=224, dynamic_image_size=False,
                 use_thumbnail=False, max_num=6):
        self.datas = open(test).readlines()
        self.prompt = prompt
        self.input_size = input_size
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.max_num = max_num
        self.transform = build_transform(is_train=False, input_size=input_size)

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        data = json.loads(self.datas[idx].strip())
        image = data['image']
        text = data['text']
        bbox = data['abs_position']
        gt_bbox = list(map(int, bbox.strip('<>').split(',')))
        image_path = data["image"].replace("/home/test/test03","/home/test/test12")

        image = Image.open(image_path).convert('RGB')
        w, h = image.width, image.height
        gt_bbox[0] = int(gt_bbox[0]/w*1000)
        gt_bbox[1] = int(gt_bbox[1]/h*1000)
        gt_bbox[2] = int(gt_bbox[2]/w*1000)
        gt_bbox[3] = int(gt_bbox[3]/h*1000)

        if self.dynamic_image_size:
            images = dynamic_preprocess(image, image_size=self.input_size,
                                        use_thumbnail=self.use_thumbnail,
                                        max_num=self.max_num)
        else:
            images = [image]
        pixel_values = [self.transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)

        return {
            'pixel_values': pixel_values,
            'question': self.prompt.format(gt_bbox),
            'target': text,
            'hw': (h, w),
        }


class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)
    
def is_click(predict_bbox, target_bbox):
    """
    Args:
        predict_bbox: Tensor of shape [bs, 2], format [x, y] (coordinates of predicted points).
        target_bbox: Tensor of shape [bs, 4], format [x1, y1, x2, y2] (bounding box coordinates).
    Returns:
        Tensor of shape [bs], where each element is 1 if the point is inside the box, else 0.
    """
    # Extract coordinates
    x, y = predict_bbox[:, 0], predict_bbox[:, 1]
    x1, y1, x2, y2 = target_bbox[:, 0], target_bbox[:, 1], target_bbox[:, 2], target_bbox[:, 3]

    # Check if point is inside the box (inclusive)
    inside_x = (x >= x1) & (x <= x2)
    inside_y = (y >= y1) & (y <= y2)
    inside = inside_x & inside_y

    # Convert boolean to 1/0
    return inside.int()



def evaluate_chat_model():
    random.seed(args.seed)
    summaries = []
    dataset_dir = "your/path/to/the/dataset"
    dataset = ChineseDataset(
        test=dataset_dir,
        prompt=prompt,
        input_size=image_size,
        dynamic_image_size=args.dynamic,
        use_thumbnail=use_thumbnail,
        max_num=args.max_num
    )
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        sampler=InferenceSampler(len(dataset)),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=partial(collate_fn, tokenizer=tokenizer),
    )

    outputs = []
    for _, (pixel_values, questions, targets, hws) in enumerate(tqdm(dataloader)):
        pixel_values = pixel_values.to(torch.bfloat16).cuda()
        generation_config = dict(
            num_beams=args.num_beams,
            max_new_tokens=100,
            min_new_tokens=1,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
        )
        pred = model.chat(
            tokenizer=tokenizer,
            pixel_values=pixel_values,
            question=questions[0],
            generation_config=generation_config,
            verbose=True
        )
        answers = [pred]

        for target, answer in zip(targets, answers):
            outputs.append({
                'target': target,
                'answer': answer,
            })

        torch.distributed.barrier()

        world_size = torch.distributed.get_world_size()
        merged_outputs = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(merged_outputs, outputs)

        merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]

        if torch.distributed.get_rank() == 0:
            time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
            results_file = f'{time_prefix}.json'
            results_file = os.path.join(args.out_dir, results_file)
            json.dump(merged_outputs, open(results_file, 'w'))

            correct = total_cnt = 0
            for i, output in enumerate(merged_outputs):
                print(output)
                if output['answer'].strip() == output['target'].strip():
                    correct += 1
                total_cnt += 1

            print(f'Precision @ 1: {correct / total_cnt} \n')
            summaries.append([args.checkpoint, f'Precision @ 1: {correct / total_cnt} \n'])

        torch.distributed.barrier()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--datasets', type=str, default='grounding')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--num-beams', type=int, default=1)
    parser.add_argument('--out-dir', type=str, default='results')
    parser.add_argument('--sample', type=bool, default=False)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dynamic', action='store_true')
    parser.add_argument('--max-num', type=int, default=6)
    parser.add_argument('--load-in-8bit', action='store_true')
    parser.add_argument('--load-in-4bit', action='store_true')
    parser.add_argument('--auto', action='store_true')
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)

    args.datasets = args.datasets.split(',')
    print('datasets:', args.datasets)
    assert args.batch_size == 1, 'Only batch size 1 is supported'

    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )

    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

    PATTERN = re.compile(r'\[\[(.*?),(.*?),(.*?),(.*?)\]\]')
    model, tokenizer = load_model_and_tokenizer(args)
    image_size = model.config.force_image_size or model.config.vision_config.image_size
    use_thumbnail = model.config.use_thumbnail
    prompt = '''你是一个GUI组件文字识别的专家，擅长根据组件的边界框（bounding box）描述输出对应的文字。组件的边界框<ref>{}</ref>,你应该输出对应的文字！！'''

    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    if total_params > 20 or args.dynamic:
        args.num_beams = 1
        print(f'[test] total_params: {total_params}B, use num_beams: {args.num_beams}')
    else:
        print(f'[test] total_params: {total_params}B')

    evaluate_chat_model()
