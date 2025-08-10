import cv2
import numpy as np
from rubikvision.cube_solver import CubePlanner
from rubikvision.utils import find_webcam_index

def run_video(video_stream, cube_planner, plot_bounding_box=False, plot_projection=True, plot_cube_state=True, rotate_img=False):
    cap = cv2.VideoCapture(video_stream)
    while True:
        ret, image = cap.read()
        assert ret
        cube_planner.init_image(image, rotate_img=rotate_img)

        if rotate_img:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

        cube_planner.plot(image,
                          plot_bounding_box=plot_bounding_box,
                          plot_projection=plot_projection,
                          plot_cube_state=plot_cube_state)



        cv2.imshow('img',image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # K: C922 Pro Stream Webcam camera intrinsic https://www.calibdb.net/
    cube_planner = CubePlanner(K=np.array([[632.11326486, 0., 316.16980761],
                                            [0., 630.54696352, 233.72252151],
                                            [0., 0., 1.]]))
    video_stream = find_webcam_index("C922 Pro Stream Webcam")
    run_video(video_stream=video_stream,
              cube_planner=cube_planner,
              plot_bounding_box=False,
              plot_projection=True,
              plot_cube_state=True,
              rotate_img=True)
