import colorsys
import os
import random
import warnings

import clip
import cv2
import numpy as np
from fastsam import FastSAMPrompt, FastSAM

from rubikvision.cube_pose import get_square_contours, calculate_midpoint, shift_points


def get_model():
    model_path = os.path.join(os.path.dirname(__file__), 'weights','FastSAM.pt')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = FastSAM(model_path)
    return model


def get_clip_model_preprocess(device, model='ViT-B/32'):
    clip_model, preprocess = clip.load(model, device=device)
    return clip_model, preprocess


class FastTextPrompt(FastSAMPrompt):
    def __init__(self, clip_model, preprocess, *args, **kwargs):
        super(FastTextPrompt, self).__init__(*args, **kwargs)
        self.clip_model = clip_model
        self.preprocess = preprocess

    def text_prompt(self, text):
        if self.results == None:
            return []
        format_results = self._format_results(self.results[0], 0)
        cropped_boxes, cropped_images, not_crop, filter_id, annotations = self._crop_image(format_results)

        scores = self.retrieve(self.clip_model, self.preprocess, cropped_boxes, text, device=self.device)
        max_idx = scores.argsort()
        max_idx = max_idx[-1]
        max_idx += sum(np.array(filter_id) <= int(max_idx))
        return np.array([annotations[max_idx]['segmentation']])

def resize_with_aspect_ratio(image, new_width=None, new_height=None):
    original_height, original_width = image.shape[:2]
    if new_width is None and new_height is None:
        return image
    if new_width is not None and new_height is not None:
        pass
    if new_width is None:
        ratio = new_height / original_height
        new_width = int(original_width * ratio)
    elif new_height is None:
        ratio = new_width / original_width
        new_height = int(original_height * ratio)
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    return resized_image

class CubeSegmentation:
    def __init__(self, device, clip_model='ViT-B/32'):
        self.device = device
        self.model = get_model()
        self.clip_model, self.preprocess = get_clip_model_preprocess(device, model=clip_model)
        self._last_prompt = None


    def segment_everything(self, frame):
        everything_results = self.model(frame, device=self.device, retina_masks=True, imgsz=512, conf=0.4, iou=0.9, verbose=False)
        prompt_process = FastTextPrompt(self.clip_model, self.preprocess, frame, everything_results, device=self.device)
        self._last_prompt = prompt_process
        ann = prompt_process.everything_prompt()
        return ann

    def segment_cube(self, frame):
        everything_results = self.model(frame, device=self.device, retina_masks=True, imgsz=640, conf=0.4, iou=0.9, verbose=False)
        prompt_process = FastTextPrompt(self.clip_model, self.preprocess, frame, everything_results, device=self.device)
        self._last_prompt = prompt_process
        ann = prompt_process.text_prompt(text='Cube, Box, Quadrat, 3x3, colored, colors')
        annotation = ann[0]
        return annotation

    def detect_cube(self, frame, new_width=0):
        if new_width == 0:
            ann = self.segment_cube(frame)
            box = self._extract_bounding_box(ann)
        else:
            height, width = frame.shape[:2]
            frame_new = resize_with_aspect_ratio(frame, new_width=new_width)
            ann_new = self.segment_cube(frame_new)
            box_new = self._extract_bounding_box(ann_new)
            ann = resize_with_aspect_ratio(ann_new.astype(np.float32), width, height).astype(bool)
            factor = frame.shape[0] / frame_new.shape[0]
            box = {k: int(v * factor) for k,v in box_new.items()}
        return box, ann

    def _extract_bounding_box(self, segmentation):
        # Find the indices where the segmentation is True
        true_indices = np.argwhere(segmentation)
        if true_indices.size == 0:
            return None, None, None, None  # No object detected
        min_y, min_x = true_indices.min(axis=0)
        max_y, max_x = true_indices.max(axis=0)
        midpoint_x = (min_x + max_x) // 2
        midpoint_y = (min_y + max_y) // 2
        width = max_x - min_x
        height = max_y - min_y
        return {'x': midpoint_x, 'y': midpoint_y, 'width': width, 'height': height}

    def get_midpoints(self, img, box):
        roi, x_min, y_min = extract_cube(img, box)
        annotation = self.segment_everything(roi)
        if annotation == []:
            print('COULD NOT SEGMENT CUBE')
            return []
        contours = get_square_contours(annotation)
        mid_points = [calculate_midpoint(cnt) for cnt in contours]
        mid_points = shift_points(mid_points, shift=(x_min, y_min))
        return mid_points

class CubeSegmentationRoboflow(CubeSegmentation):
    def __init__(self, *args, **kwargs):
        super(CubeSegmentationRoboflow, self).__init__(*args, **kwargs)
        from roboflow import Roboflow
        api_key = os.environ.get('ROBO_KEY', '')
        assert api_key is not None, 'enter api key here'
        rf = Roboflow(api_key=api_key)
        project = rf.workspace().project("my_detect_rubik")
        model_yolo = project.version(1).model
        self.model_yolo = model_yolo

    def detect_cube(self, frame, new_width=0):
        if new_width == 0:
            response = self.model_yolo.predict(frame).json()
            if response['predictions'] == []:
                return None, None
            box = response['predictions'][0]
        else:
            frame_new = resize_with_aspect_ratio(frame, new_width=new_width)
            response = self.model_yolo.predict(frame_new).json()
            if response['predictions'] == []:
                return None, None
            box = response['predictions'][0]
            factor = frame.shape[0] / frame_new.shape[0]
            for key in ['x', 'y', 'width', 'height']:
                box[key] = box[key] * factor
        return box, None

def extract_cube(img, box, EPS=10):
    y_min, y_max = int(box['y'] - box['height'] / 2 - EPS), int(box['y'] + box['height'] / 2 + EPS)
    x_min, x_max = int(box['x'] - box['width'] / 2 - EPS), int(box['x'] + box['width'] / 2 + EPS)
    y_min  = max(0, y_min)
    x_min  = max(0, x_min)
    roi = img[y_min:y_max,
              x_min:x_max]
    return roi, x_min, y_min

def draw_bounding_box(image, pred, color=(0, 0, 255)):
    top_left = (int(pred['x'] - pred['width'] / 2), int(pred['y'] - pred['height'] // 2))
    bottom_right = (int(pred['x'] + pred['width'] // 2), int(pred['y'] + pred['height'] // 2))
    cv2.rectangle(image, top_left, bottom_right, color, 1)


def highlight_points(image, points, projections, ):
    color_original = (0, 255, 0)  # Green for original points
    color_estimate = (0, 0, 255)  # Red for estimated points
    circle_radius = 3
    x_size = 1
    for point in points:
        # Draw the original point as a circle
        cv2.circle(image, (int(point[0]), int(point[1])), circle_radius, color_original, 2)
    for estimate in  projections:
        cv2.line(image, (int(estimate[0] - x_size), int(estimate[1] - x_size)),
                 (int(estimate[0] + x_size), int(estimate[1] + x_size)), color_estimate, 2)
        cv2.line(image, (int(estimate[0] - x_size), int(estimate[1] + x_size)),
                 (int(estimate[0] + x_size), int(estimate[1] - x_size)), color_estimate, 2)


def draw_cube_adaptive_edges(img, imgpts, thickness=1):
    num_samples = 30
    imgpts = np.int32(imgpts).reshape(-1, 2)
    max_y, max_x, _ = img.shape
    for i in range(8):
        for j in [(i + 1) % 4 + (i // 4) * 4, (i + 4) % 8]:
            start = tuple(imgpts[i])
            end = tuple(imgpts[j])
            # Get the average color along the edge
            x = np.linspace(start[0], end[0], num_samples)
            y = np.linspace(start[1], end[1], num_samples)
            x = np.maximum(np.minimum(max_x-1, x),0).astype(int)
            y = np.maximum(np.minimum(max_y-1, y),0).astype(int)
            edge_color = np.mean(img[y, x], axis=0)
            # Choose a contrasting color
            line_color = 255 - edge_color
            cv2.line(img, start, end, tuple(line_color), thickness)
    return img

def get_colors(num_colors: int):
    """Gets colormap for points."""
    colors = []
    for i in np.arange(0.0, 360.0, 360.0 / num_colors):
        hue = i / 360.0
        lightness = (50 + np.random.rand() * 10) / 100.0
        saturation = (90 + np.random.rand() * 10) / 100.0
        color = colorsys.hls_to_rgb(hue, lightness, saturation)
        colors.append(
            (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
        )
    random.shuffle(colors)
    return colors

