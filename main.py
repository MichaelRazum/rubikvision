import subprocess
import cv2
import threading
import queue
import numpy as np
import torch

from cube_pose import estimate_cube_pose, get_cube_edges
from cube_detection import CubeSegmentation, draw_cube_adaptive_edges, draw_bounding_box, highlight_points


def find_webcam_index(device_name):
    command = "v4l2-ctl --list-devices"
    output = subprocess.check_output(command, shell=True, text=True)
    devices = output.split('\n\n')

    for device in devices:
        #print(device)
        if device_name in device:
            lines = device.split('\n')
            for line in lines:
                if "video" in line:
                    parts = line.split()
                    for part in parts:
                        if part.startswith('/dev/video'):
                            return (part)

def estimate_cube_pose_que(K, dist_coeffs, cube_seg: CubeSegmentation, frame_queue, result_queue):
    while True:
        frame = frame_queue.get()
        if frame is None:
            break
        box, _ = cube_seg.detect_cube(frame)
        rvec, tvec, midpoints, projection = estimate_cube_pose(cube_seg=cube_seg,box=box, img=frame, K=K, dist_coeffs=dist_coeffs)
        if rvec is not None:
            result_queue.put((box, tvec, rvec, midpoints, projection))

def highlight_cube(frame, tvec, rvec):
    cube_points = get_cube_edges()
    imgpts, _ = cv2.projectPoints(cube_points, rvec, tvec, K, dist_coeffs)
    img = draw_cube_adaptive_edges(frame, imgpts)
    return img



def run_video(video_stream, K, plot_bounding_box=False, plot_projection=True):
    cap = cv2.VideoCapture(video_stream)

    frame_queue = queue.Queue(maxsize=1)
    result_queue = queue.Queue()

    # Start the processing thread
    processing_thread = threading.Thread(target=estimate_cube_pose_que, args=(K, dist_coeffs, cube_seg, frame_queue, result_queue))
    processing_thread.start()

    last_box, last_tvec, last_rvec, last_midponts, last_proj = None, None, None, None, None
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Try to get the latest result without blocking
        try:
            last_box, last_tvec, last_rvec,  last_midponts, last_proj = result_queue.get_nowait()
        except queue.Empty:
            pass

        # Highlight the cube using the last known position
        if last_tvec is not None and last_rvec is not None:
            frame = highlight_cube(frame, last_tvec, last_rvec)

            if plot_bounding_box:
                draw_bounding_box(image=frame, pred=last_box,  color=(0, 0, 255))

            if plot_projection:
                if last_midponts is not None and last_proj is not None:
                    highlight_points(points=last_midponts, projections=last_proj, image=frame)

        cv2.imshow('Cube Highlighter', frame)
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

    # C922 Pro Stream Webcam camera intrinsic
    # https://www.calibdb.net/
    K = np.array([[632.11326486, 0., 316.16980761],
                  [0., 630.54696352, 233.72252151],
                  [0., 0., 1.]])

    dist_coeffs = np.zeros((4, 1))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'loading cube seg, with {device=}')
    cube_seg = CubeSegmentation(device=device)
    video_stream = find_webcam_index("C922 Pro Stream Webcam")
    run_video(video_stream=video_stream, K=K, plot_bounding_box=False, plot_projection=True)
