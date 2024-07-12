import clip
import cv2
import numpy as np
from fastsam import FastSAMPrompt, FastSAM

def get_model():
    model_path = '/home/mira/Projects/FastSAM/weights/FastSAM.pt'
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


class CubeSegmentation:
    def __init__(self, device, clip_model='ViT-B/32'):
        self.device = device
        self.model = get_model()
        self.clip_model, self.preprocess = get_clip_model_preprocess(device, model=clip_model)
        self._last_prompt = None


    def __call__(self, frame, segment_everything=False):
        everything_results = self.model(frame, device=self.device, retina_masks=True, imgsz=640, conf=0.4, iou=0.9, )
        prompt_process = FastTextPrompt(self.clip_model, self.preprocess, frame, everything_results, device=self.device)
        self._last_prompt = prompt_process
        if not segment_everything:
            ann = prompt_process.text_prompt(text='Cube, Box, Quadrat, 3x3, colored, colors')
            annotation = ann[0]
            return annotation
        else:
            ann = prompt_process.everything_prompt()
            return ann

    def segment_cube(self, frame):
        everything_results = self.model(frame, device=self.device, retina_masks=True, imgsz=640, conf=0.4, iou=0.9, )
        prompt_process = FastTextPrompt(self.clip_model, self.preprocess, frame, everything_results, device=self.device)
        self._last_prompt = prompt_process
        ann = prompt_process.text_prompt(text='Cube, Box, Quadrat, 3x3, colored, colors')
        annotation = ann[0]
        return annotation

    def detect_cube(self, frame):
        annotation = self.segment_cube(frame)
        bounding_box = self._extract_bounding_box(annotation)
        return bounding_box, annotation

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

def extract_cube(img, box, EPS=10):
    y_min, y_max = int(box['y'] - box['height'] / 2 - EPS), int(box['y'] + box['height'] / 2 + EPS)
    x_min, x_max = int(box['x'] - box['width'] / 2 - EPS), int(box['x'] + box['width'] / 2 + EPS)
    roi = img[y_min:y_max,
              x_min:x_max]
    return roi, x_min, y_min

def draw_bounding_box(image, pred, color=(0, 0, 255)):
    top_left = (int(pred['x'] - pred['width'] / 2), int(pred['y'] - pred['height'] // 2))
    bottom_right = (int(pred['x'] + pred['width'] // 2), int(pred['y'] + pred['height'] // 2))
    cv2.rectangle(image, top_left, bottom_right, color, 1)

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