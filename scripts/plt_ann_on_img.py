# plot the annotation file and image path and plot the boxes on img
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
METAINFO = {
	'classes': ('CFCBK', 'FCBK', 'Zigzag'),
	'palette': [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
}

add_score = False

def draw_annotations(image, ann_file):
	"""Draws rotated bounding boxes from a DOTA annotation file using specified colors."""
	img = image.copy()
	if not os.path.exists(ann_file):
		print(f"Annotation file {ann_file} not found.")
		return img
	class_to_color = dict(zip(METAINFO['classes'], METAINFO['palette']))
	with open(ann_file, 'r') as f:
		lines = f.readlines()
	for line in lines:
		values = line.strip().split()
		if len(values) < 9:
			continue
		points = np.array(list(map(float, values[:8]))).reshape((4, 2)).astype(int)
		label = values[8]
		score = values[9]
		color = class_to_color.get(label, (255, 255, 255))		
		if add_score:
			label = label + ':' +  score
		thickness = 1
		cv2.polylines(img, [points], isClosed=True, color=color, thickness=thickness)
		cv2.putText(img, label, (points[0][0], points[0][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
	return img


def annotate_directory(image_dir, ann_dir, out_dir, save=True, plot=False):
	"""Draws rotated bounding boxes from DOTA annotation files on images in a directory."""
	if not os.path.exists(out_dir):
		os.makedirs(out_dir, exist_ok=True)
	image_files = os.listdir(image_dir)
	image_files = sorted(image_files)
	num_images = len(image_files)
	for i, image_name in enumerate(image_files):
		print(f'Processing image {i + 1}/{num_images}...')
		image_path = os.path.join(image_dir, image_name)
		ann_file = os.path.join(ann_dir, f'{os.path.splitext(image_name)[0]}.txt')
		image = cv2.imread(image_path)
		if image is None:
			print(f"Image {image_path} not found.")
			continue
		annotated_img = draw_annotations(image, ann_file)
		if save:
			out_path = os.path.join(out_dir, image_name)
			success = cv2.imwrite(out_path, annotated_img)
			if not success:
				print(f"Failed to save annotated image to {out_path}")
		if plot:
			plt.figure(figsize=(10, 10))
			plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
			plt.axis('off')
			plt.show()


image_dir='data/m0/val/images'
ann_dir='results/m0/val/annfiles'
out_dir='results/m0/val/annotated'
annotate_directory(image_dir, ann_dir, out_dir, save=True, plot=False)
