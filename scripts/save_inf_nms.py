import os
import cv2
import mmcv
import torch
import numpy as np
import matplotlib.pyplot as plt
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmdet.apis import inference_detector, init_detector
from mmrotate.visualization import RotLocalVisualizer
from mmcv.ops import nms_rotated
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


def apply_rotated_nms(bboxes, scores, labels, nms_iou_threshold=0.1):
    """Applies Rotated Non-Maximum Suppression (R-NMS) to filter overlapping detections."""
    if len(bboxes) == 0:
        return [], [], []

    # Convert numpy arrays to tensors
    bboxes = torch.tensor(bboxes, dtype=torch.float32)
    scores = torch.tensor(scores, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.int64)

    # Apply rotated NMS
    _, keep_indices = nms_rotated(bboxes, scores, nms_iou_threshold)
    # print(keep_indices)

    # Convert tensors back to numpy
    bboxes = bboxes[keep_indices].numpy()
    scores = scores[keep_indices].numpy()
    labels = labels[keep_indices].numpy()

    return bboxes, scores, labels



def save_inference(image_dir, model, inf_dir, visualizer, class_mapping, confidence_threshold=0.25, nms_iou_threshold=0.1, standardize_points=False, img_size=640, save_images=False, save_ann=False):
    os.makedirs(inf_dir, exist_ok=True)
    image_files = os.listdir(image_dir)
    image_files = sorted(image_files)
    num_images = len(image_files)

    for i, image_name in enumerate(image_files):
        print(f'Processing image {i + 1}/{num_images}: {image_name}')
        image_path = os.path.join(image_dir, image_name)
        result = inference_detector(model, image_path)

        bboxes = result.pred_instances.bboxes.cpu().numpy()
        scores = result.pred_instances.scores.cpu().numpy()
        labels = result.pred_instances.labels.cpu().numpy()

        # Filter detections based on confidence threshold
        mask = scores >= confidence_threshold
        bboxes, scores, labels = bboxes[mask], scores[mask], labels[mask]

        bboxes, scores, labels = apply_rotated_nms(bboxes, scores, labels, nms_iou_threshold=nms_iou_threshold)
        
        # Save the visualized test images with detections
        if save_images:
            inf_img_dir = os.path.join(inf_dir, 'images')
            os.makedirs(inf_img_dir, exist_ok=True)
            visualizer.add_datasample(
                image_name,
                mmcv.imread(image_path),
                result,
                show=False,
                draw_gt=False,
                out_file=os.path.join(inf_img_dir, image_name)
            )

        # Save annotations in DOTA format
        if save_ann:
            inf_ann_dir = os.path.join(inf_dir, 'annfiles')
            os.makedirs(inf_ann_dir, exist_ok=True)
            ann_file = os.path.join(inf_ann_dir, f'{os.path.splitext(image_name)[0]}.txt')
            with open(ann_file, 'w') as f:
                for bbox, score, label in zip(bboxes, scores, labels):
                    # Convert (cx, cy, w, h, angle) to DOTA format
                    cx, cy, w, h, angle = bbox  # Rotated bbox format: (center_x, center_y, width, height, angle)
                    points = cv2.boxPoints(((cx, cy), (w, h), np.degrees(angle)))  # Get 4 corner points
                    points = points.flatten()

                    if standardize_points:
                        points = points / img_size

                    label_name = class_mapping.get(label, 'Unknown')
                    f.write(f"{' '.join(map(str, points))} {label_name} {score:.2f}\n")

    print('Done.')


# Configuration and checkpoint files
config_file = 'configs-mine/rhino/rhino_phc_haus-4scale_r50_2xb2-36e_brickkilns.py'

checkpoint_file = 'work_dirs/rhino_phc_haus-4scale_r50_2xb2-36e_bihar/epoch_36.pth'
image_dir = 'data/bihar/val/images'
inf_dir = 'results/train_bihar_test_bihar'
save_images = False
save_ann = True
standardize_points = False
img_height = 640
img_width = 640
replace_class_with_label = False
class_mapping = {0: "CFCBK", 1: "FCBK", 2: "Zigzag"}
nms_iou_threshold = 0.33
confidence_threshold = 0.25


# Load the configuration file
cfg = Config.fromfile(config_file)
init_default_scope(cfg.get('default_scope', 'mmrotate'))
# Initialize the model
model = init_detector(cfg, checkpoint_file, device='cuda:0')
# Initialize the visualizer
visualizer = RotLocalVisualizer()


# Call the function
save_inference(image_dir, model, inf_dir, visualizer, class_mapping, confidence_threshold, nms_iou_threshold, standardize_points, img_height, save_images, save_ann)
