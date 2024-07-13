import cv2
import pytest
import os

from matplotlib import pyplot as plt

import numpy as np
import math

from cube_detection import draw_bounding_box

def fit_hexagon(contour, target_points=6):
    perimeter = cv2.arcLength(contour, True)
    epsilon = 0.02
    for _ in range(100):
        approx = cv2.approxPolyDP(contour, epsilon * perimeter, True)
        if len(approx) == target_points:
            return approx
        elif len(approx) > target_points:
            epsilon += 0.01
        else:
            epsilon -= 0.001
        if epsilon <= 0 or epsilon > 1:
            raise ValueError("Unable to find target points. Check your contour.")
    return approx



def plot_compare_img_detection(img, annotation, pred_1, pred_2):
    mask = np.zeros_like(img, dtype=np.uint8)
    mask[annotation == 1] = [0, 255, 0]  # Green color for annotation
    # Set transparency level
    alpha = 0.3
    # Overlay mask on frame
    overlay = img.copy()
    cv2.addWeighted(mask, alpha, overlay, 1 - alpha, 0, overlay)

    draw_bounding_box(overlay, pred_1, color=(0, 0, 255))  # Red color
    draw_bounding_box(overlay, pred_2, color=(255, 0, 0))  # Blue color

    cv2.imshow('Segmentation with Bounding Boxes', overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


@pytest.mark.parametrize('img_id', range(13))
def test_find_cube_compare_roboflow_and_fastSAM(img_id, model_yolo, cube_seg, datadir, plot=False):
    if img_id == 9:
        pytest.xfail("Roboflow model doesn't return an accurate detection")

    img_path = os.path.join(datadir, f'cubepic_{img_id}.jpg')
    img = cv2.imread(img_path)

    # Roboflow Rubik Cube Object detection
    response = model_yolo.predict(img).json()
    pred_1 = response['predictions'][0]

    # Fast SAM Cube Detection
    pred_2, annotation =  cube_seg.detect_cube(img)

    PIXEL_TOL_ERR = 20
    for key in pred_2:
        assert math.isclose(pred_2[key], pred_1[key], abs_tol=PIXEL_TOL_ERR)
    # *** PLOT ***
    if plot:
        plot_compare_img_detection(img, annotation, pred_1, pred_2)



@pytest.mark.parametrize('img_id', range(13))
def test_find_hexagon(img_id, cube_seg, datadir, plot=False):
    img_path = os.path.join(datadir, f'cubepic_{img_id}.jpg')
    img = cv2.imread(img_path)
    annotation = cube_seg(img)
    contours, _ = cv2.findContours(annotation.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = contours[0]
    fitted_hexagon = fit_hexagon(contour)
    assert 5<=len(fitted_hexagon) <=6
    # PLOT
    if plot:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img_rgb)
        iterative_hexagon = fitted_hexagon.squeeze()
        for point in iterative_hexagon:
            x, y = point
            ax.plot(x, y, 'ro', markersize=10)  # 'ro' means red circle
        hexagon_closed = np.vstack(
            (iterative_hexagon, iterative_hexagon[0]))  # Add the first point to the end to close the shape
        xs, ys = hexagon_closed[:, 0], hexagon_closed[:, 1]
        ax.plot(xs, ys, 'r-', linewidth=2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('Iterative ApproxPolyDP Hexagon')
        plt.show()