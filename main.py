import os
import cv2
import threading
import queue

import kociemba
import numpy as np
import torch

from color_classifier import ColorClassiferKmeans, ColorClassfierEnsemble
from cube_pose import estimate_cube_pose, get_cube_edges, get_surfaces_Q1_Q2_Q3
from cube_detection import CubeSegmentation, draw_cube_adaptive_edges, draw_bounding_box, highlight_points
from cube_solver import CubeSolver
from utils import find_webcam_index


def get_auto_index(dataset_dir='image_out', dataset_name_prefix = '', data_suffix = 'mp4'):
    max_idx = 1000
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    for i in range(max_idx+1):
        if not os.path.isfile(os.path.join(dataset_dir, f'{dataset_name_prefix}cubepic_{i}.{data_suffix}')):
            return i
    raise Exception(f"Error getting auto index, or more than {max_idx} episodes")

def estimate_cube_pose_que(K, dist_coeffs, cube_seg: CubeSegmentation, frame_queue, result_queue):
    while True:
        frame = frame_queue.get()
        if frame is None:
            break
        box, _ = cube_seg.detect_cube(frame,new_width=WIDTH_CUBE_DETECT)
        if box is None:
            print('could not detect cube')
            continue
        mid_points = cube_seg.get_midpoints(frame, box)

        rvec, tvec, midpoints, projection_inliners = estimate_cube_pose(mid_points=mid_points, K=K, dist_coeffs=dist_coeffs)
        if rvec is not None:
            result_queue.put((box, tvec, rvec, midpoints, projection_inliners, frame))

        if SAVE_IMAGES and rvec is None or projection_inliners is None or len(mid_points) < 8 or len(projection_inliners)<8 :
            print('saving image')
            i = get_auto_index(dataset_dir='image_out/', data_suffix='jpg')
            fname = 'image_out/cubepic_' + str(i) + '.jpg'
            cv2.imwrite(fname, frame)

def highlight_cube(frame, tvec, rvec):
    cube_points = get_cube_edges()
    imgpts, _ = cv2.projectPoints(cube_points, rvec, tvec, K, dist_coeffs)
    img = draw_cube_adaptive_edges(frame, imgpts)
    return img


def run_video(video_stream, K, plot_bounding_box=False, plot_projection=True, plot_cube_state=True, rotate_img=False):
    cap = cv2.VideoCapture(video_stream)

    frame_queue = queue.Queue(maxsize=1)
    result_queue = queue.Queue()

    processing_thread = threading.Thread(target=estimate_cube_pose_que, args=(K, dist_coeffs, cube_seg, frame_queue, result_queue))
    processing_thread.start()

    last_box, last_tvec, last_rvec, last_midponts, last_proj, last_frame = None, None, None, None, None, None
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if rotate_img:
            frame =  cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        # Try to get the latest result without blocking
        try:
            last_box, last_tvec, last_rvec,  last_midponts, last_proj, last_frame  = result_queue.get_nowait()
        except queue.Empty:
            pass

        # Try to set cube state
        try:
            success, proj2d_s = get_surfaces_Q1_Q2_Q3(rvec=last_rvec,
                                                      tvec=last_tvec,
                                                      K=K,
                                                      dist_coeffs=dist_coeffs,
                                                      inliners=last_proj)
            if success:
                cube_solver.update_cube(img=last_frame, proj2d_s=proj2d_s)
        except Exception as e:
            pass

        # Highlight the cube using the last known position
        image = frame.copy()
        if last_tvec is not None and last_rvec is not None:
            image = highlight_cube(image, last_tvec, last_rvec)

            if plot_bounding_box:
                draw_bounding_box(image=image, pred=last_box,  color=(0, 0, 255))

            if plot_projection:
                if last_midponts is not None and last_proj is not None:
                    highlight_points(points=last_midponts, projections=last_proj, image=image)

            if plot_cube_state:
                if cube_solver.solution is not None:
                    print(cube_solver.solution)
                cube_solver.cube_state.plot(image)

        cv2.imshow('img',image)

        while True:
            try:
                frame_queue.get_nowait()  # Remove old frame if exists
            except queue.Empty:
                break
        frame_queue.put(frame.copy())

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    frame_queue.put(None)  # Signal the processing thread to stop
    processing_thread.join()

if __name__ == "__main__":
    # FastSAM https://huggingface.co/spaces/An-619/FastSAM
    # Used weights
    # https://huggingface.co/spaces/An-619/FastSAM/resolve/main/weights/FastSAM.pt

    #### CAMERA INTRINSICS ####
    # C922 Pro Stream Webcam camera intrinsic
    # https://www.calibdb.net/
    K = np.array([[632.11326486, 0., 316.16980761],
                  [0., 630.54696352, 233.72252151],
                  [0., 0., 1.]])
    dist_coeffs = np.zeros((4, 1))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'loading cube seg, with {device=}')

    # lower value for faster cube detection
    WIDTH_CUBE_DETECT = 320

    # SAVE IMAGES for bad pose estimations
    SAVE_IMAGES = False


    cube_seg = CubeSegmentation(device=device)
    color_cls = ColorClassiferKmeans().load()
    cube_solver = CubeSolver(color_cls)
    # cube_seg = CubeSegmentationRoboflow(device=device)
    video_stream = find_webcam_index("C922 Pro Stream Webcam")
    run_video(video_stream=video_stream, K=K, plot_bounding_box=False, plot_projection=True, plot_cube_state=True, rotate_img=True)
