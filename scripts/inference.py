# Description: This script is used to perform inference on the validation set and save the results in the DOTA format or Supervision format after applying NMS.

import os
import cv2
import mmcv
import torch
import numpy as np
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmdet.apis import inference_detector, init_detector
from mmrotate.visualization import RotLocalVisualizer
from mmcv.ops import nms_rotated
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


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


def save_inference(image_dir, model, inf_dir, class_mapping, confidence_threshold=0.25, apply_nms=True, nms_iou_threshold=0.1, standardize_points=False, img_size=640, save_images=False, save_ann=False, save_ann_format='dota'):
    os.makedirs(inf_dir, exist_ok=True)
    image_files = os.listdir(image_dir)
    image_files = sorted(image_files)
    num_images = len(image_files)

    visualizer = RotLocalVisualizer()

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

        if apply_nms:
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
                draw_pred=True,
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

                    if replace_class_with_label:
                        label = class_mapping.get(label, 'Unknown')
                    else:
                        label = str(label)
                    
                    if save_ann_format == 'dota':
                        f.write(f"{' '.join(map(str, points))} {label} {score}\n")
                    elif save_ann_format == 'supervision':
                        f.write(f"{label} {' '.join(map(str, points))} {score}\n")

    print('Done.')


# Configuration files
model_configs = [
    {
        # Combined Data to 5 state bihar
        'train': 'Combined Data',
        'test': 'Bihar',
        'backbone': 'resnet50',
        'head': 'rhino',
        'config_file': 'configs-mine/rhino-resnet/rhino_phc_haus-4scale_r50_2xb2-36e_combined.py',
        'checkpoint_file': '/home/shardul.junagade/my-work/domain-adaptation-brick-kilns/RHINO/work_dirs/rhino_phc_haus-4scale_r50_2xb2-36e_combined/best_model_on_m0.pth',
        'val_dir': '/home/shardul.junagade/my-work/domain-adaptation-brick-kilns/RHINO/data/5states/bihar',
        'inf_dir': '/home/shardul.junagade/my-work/domain-adaptation-brick-kilns/RHINO/results-resnet50/train_combined_test_5states/bihar',
        'img_height': 640,
        'epoch': 21,
    },
    {
        # Combined Data to 5 state haryana
        'train': 'Combined Data',
        'test': 'Haryana',
        'backbone': 'resnet50',
        'head': 'rhino',
        'config_file': 'configs-mine/rhino-resnet/rhino_phc_haus-4scale_r50_2xb2-36e_combined.py',
        'checkpoint_file': '/home/shardul.junagade/my-work/domain-adaptation-brick-kilns/RHINO/work_dirs/rhino_phc_haus-4scale_r50_2xb2-36e_combined/best_model_on_m0.pth',
        'val_dir': '/home/shardul.junagade/my-work/domain-adaptation-brick-kilns/RHINO/data/5states/haryana',
        'inf_dir': '/home/shardul.junagade/my-work/domain-adaptation-brick-kilns/RHINO/results-resnet50/train_combined_test_5states/haryana',
        'img_height': 640,
        'epoch': 21,
    },
    {
        # Combined Data to 5 state punjab
        'train': 'Combined Data',
        'test': 'Punjab',
        'backbone': 'resnet50',
        'head': 'rhino',
        'config_file': 'configs-mine/rhino-resnet/rhino_phc_haus-4scale_r50_2xb2-36e_combined.py',
        'checkpoint_file': '/home/shardul.junagade/my-work/domain-adaptation-brick-kilns/RHINO/work_dirs/rhino_phc_haus-4scale_r50_2xb2-36e_combined/best_model_on_m0.pth',
        'val_dir': '/home/shardul.junagade/my-work/domain-adaptation-brick-kilns/RHINO/data/5states/punjab',
        'inf_dir': '/home/shardul.junagade/my-work/domain-adaptation-brick-kilns/RHINO/results-resnet50/train_combined_test_5states/punjab',
        'img_height': 640,
        'epoch': 21,
    },
    {
        # Combined Data to 5 state uttar pradesh
        'train': 'Combined Data',
        'test': 'Uttar Pradesh',
        'backbone': 'resnet50',
        'head': 'rhino',
        'config_file': 'configs-mine/rhino-resnet/rhino_phc_haus-4scale_r50_2xb2-36e_combined.py',
        'checkpoint_file': '/home/shardul.junagade/my-work/domain-adaptation-brick-kilns/RHINO/work_dirs/rhino_phc_haus-4scale_r50_2xb2-36e_combined/best_model_on_m0.pth',
        'val_dir': '/home/shardul.junagade/my-work/domain-adaptation-brick-kilns/RHINO/data/5states/uttar_pradesh',
        'inf_dir': '/home/shardul.junagade/my-work/domain-adaptation-brick-kilns/RHINO/results-resnet50/train_combined_test_5states/uttar_pradesh',
        'img_height': 640,
        'epoch': 21,
    },
    {
        # Combined Data to 5 state west bengal
        'train': 'Combined Data',
        'test': 'West Bengal',
        'backbone': 'resnet50',
        'head': 'rhino',
        'config_file': 'configs-mine/rhino-resnet/rhino_phc_haus-4scale_r50_2xb2-36e_combined.py',
        'checkpoint_file': '/home/shardul.junagade/my-work/domain-adaptation-brick-kilns/RHINO/work_dirs/rhino_phc_haus-4scale_r50_2xb2-36e_combined/best_model_on_m0.pth',
        'val_dir': '/home/shardul.junagade/my-work/domain-adaptation-brick-kilns/RHINO/data/5states/west_bengal',
        'inf_dir': '/home/shardul.junagade/my-work/domain-adaptation-brick-kilns/RHINO/results-resnet50/train_combined_test_5states/west_bengal',
        'img_height': 640,
        'epoch': 21,
    },
]


save_images = False
save_ann = True
save_ann_format = 'supervision'
standardize_points = True
replace_class_with_label = False
class_mapping = {0: "CFCBK", 1: "FCBK", 2: "Zigzag"}
apply_nms = True
nms_iou_threshold = 0.33
confidence_threshold = 0.01

for model_config in model_configs:
    # Load the configuration file
    cfg = Config.fromfile(model_config['config_file'])
    init_default_scope(cfg.get('default_scope', 'mmrotate'))
    # Initialize the model
    model = init_detector(cfg, model_config['checkpoint_file'], device=device)
    # Call the function
    save_inference(model_config['val_dir'] + '/images', model, model_config['inf_dir'], class_mapping, confidence_threshold, apply_nms, nms_iou_threshold, standardize_points, model_config['img_height'], save_images, save_ann, save_ann_format)
