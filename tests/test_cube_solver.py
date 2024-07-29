import kociemba
import os
import pytest
import numpy as np
import cv2

from color_classifier import ColorClassiferKmeans
from cube_pose import  estimate_cube_pose, get_surfaces_Q1_Q2_Q3
from cube_solver import CubeSolver, find_corners, find_closest_corners, align_cube_surfaces, CubeState, \
    CubeStateColorError


@pytest.fixture(scope='function')
def solver():
    clf = ColorClassiferKmeans().load()
    return CubeSolver(clf=clf)

def test_is_estimation_ok(solver):
    proj_inliner = list(range(12))
    assert solver.is_estimation_ok(proj_inliner)
    proj_inliner = list(range(4))
    assert solver.is_estimation_ok(proj_inliner) == False

def test_estimate_color(solver:CubeSolver):
    red_image = np.array([[[255, 0, 0] for _ in range(5)] for _ in range(5)], dtype=np.uint8)
    color = solver.clf.estimate_colors(red_image, [[2,2]])[0]
    assert color == 'blue'


def plot_points(img, point_dict, point_size=10, line_thickness=2):
    color_map = {
        'upper': (0, 0, 255),  # Red in BGR
        'left': (0, 255, 0),  # Green in BGR
        'front': (255, 0, 0),  # Blue in BGR
    }
    default_color = (0, 0, 0)  # Black for any other labels

    for name, points in point_dict.items():
        color = color_map.get(name, default_color)
        for point_index, point in enumerate(points):
            x, y = point
            cv2.putText(img, str(point_index), (int(x) + point_size, int(y) + point_size),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, line_thickness)

    return img


def test_find_corners():
    sample_surface = np.array([
        [0, 0], [1, 0], [2, 0],
        [0, 1], [1, 1], [2, 1],
        [0, 2], [1, 2], [2, 2]
    ])
    corners = find_corners(sample_surface)
    assert len(corners) == 4
    assert np.array_equal(corners[0], [0, 0])
    assert np.array_equal(corners[1], [2, 0])
    assert np.array_equal(corners[2], [0, 2])
    assert np.array_equal(corners[3], [2, 2])


def test_find_closest_corners():
    corners = [
        [[0, 0], [1, 0], [0, 1], [1, 1]],
        [[1.2, 0.2], [2, 0.2], [2, 0.3], [1.3, 1.2]],
        [[0.9, 1.3], [2, 3], [0.5, 2], [0.1, 1]]
    ]
    closest_triplet = find_closest_corners(corners)
    expected_triplet = ([1, 1], [1.3, 1.2], [0.9, 1.3])

    assert closest_triplet == expected_triplet, f"Expected {expected_triplet}, but got {closest_triplet}"

def test_update_cube_state():
    # colors = ['red', 'orange', 'green', 'white', 'blue', 'yellow']
    cube_state = CubeState()
    upper = ['red' for _ in range(9)]
    left = ['red' for _ in range(9)]
    front = ['red' for _ in range(9)]
    with pytest.raises(CubeStateColorError):
        cube_state.update(upper=upper, left=left, front=front)
    assert cube_state.upper == []
    assert cube_state.left == []
    assert cube_state.front == []
    assert cube_state.down == []
    assert cube_state.right == []
    assert cube_state.back == []

    # First Update
    upper = ['red' for _ in range(9)]
    left = ['orange' for _ in range(9)]
    front = ['green' for _ in range(9)]

    cube_state.update(upper=upper, left=left, front=front)
    assert cube_state.upper == upper
    assert cube_state.left == left
    assert cube_state.front == front
    assert cube_state.down == []
    assert cube_state.right == []
    assert cube_state.back == []

    # Overwrite update
    upper = ['red' for _ in range(9)]
    left = ['green' for _ in range(9)]
    front = ['blue' for _ in range(9)]
    with pytest.raises(CubeStateColorError):
        cube_state.update(upper=upper, left=left, front=front)


    # Update botton
    down = ['white' for _ in range(9)]
    left = ['green' for _ in range(9)]
    front = ['blue' for _ in range(9)]
    # known colors
    with pytest.raises(CubeStateColorError):
        cube_state.update(upper=down, left=left, front=front)


    down = ['white' for _ in range(9)]
    left = ['blue' for _ in range(9)]
    front = ['yellow' for _ in range(9)]

    cube_state.update(upper=down, left=left, front=front)
    assert cube_state.down == down
    assert cube_state.right == front
    assert cube_state.back  == left

def test_print_cube_state():
    cube_state = CubeState()
    upper= ['U' for _ in range(9)]
    left  = ['L' for _ in range(9)]
    front = ['F' for _ in range(9)]
    cube_state.update(upper=upper, left=left, front=front)

    down = ['D' for _ in range(9)]
    left = ['R' for _ in range(9)]
    right = ['B' for _ in range(9)]
    cube_state.update(upper=down, left=right, front=left)

    assert 'UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB' == cube_state.get_cube_string_notation()

@pytest.mark.parametrize('img_id', [
                                    'cube_surf.jpg',
                                    ], )
def test_map_surfaces(img_id, cube_seg, datadir, solver: CubeSolver, plot=True):
    # https://www.calibdb.net/
    # Logitech C922 PRO
    K = np.array([[632.11326486, 0., 316.16980761],
                  [0., 630.54696352, 233.72252151],
                  [0., 0., 1.]])
    dist_coeffs = np.zeros((4, 1))
    img_path = os.path.join(datadir, img_id)
    img = cv2.imread(img_path)
    box, _ = cube_seg.detect_cube(img)
    mid_points = cube_seg.get_midpoints(img, box)
    rvec, tvec,_, inliners= estimate_cube_pose(mid_points, K, dist_coeffs)
    success, proj2d_s = get_surfaces_Q1_Q2_Q3(rvec=rvec, tvec=tvec, K=K, dist_coeffs=dist_coeffs, inliners=inliners)


    alligned_surfaces = align_cube_surfaces(proj2d_s)
    map = solver.map_cube(img, alligned_surfaces)

    solver.update_cube(img, proj2d_s)

    assert map['upper'] == ['blue', 'white', 'red', 'white', 'white', 'red', 'white', 'blue', 'yellow']
    assert map['left'] == ['yellow', 'red', 'orange', 'red', 'blue', 'orange', 'white', 'yellow', 'green']
    # Todo improve classifier
    # assert map['front'] == ['green', 'yellow', 'red', 'white', 'orange', 'yellow', 'yellow', 'blue', 'orange']
    if plot:
        solver.cube_state.plot(img)
        plot_points(img,alligned_surfaces)
        cv2.imshow("Color Sampler", img)
        cv2.waitKey(0)

@pytest.mark.parametrize('cube_down, cube_upper', [['cube_down.jpg','cube_up.jpg']] )
def test_map_whole_cube(cube_down,cube_upper, cube_seg, datadir, solver: CubeSolver, plot=True):
    # https://www.calibdb.net/
    # Logitech C922 PRO
    K = np.array([[632.11326486, 0., 316.16980761],
                  [0., 630.54696352, 233.72252151],
                  [0., 0., 1.]])
    dist_coeffs = np.zeros((4, 1))

    img_path = os.path.join(datadir, cube_upper)
    img_1 = cv2.imread(img_path)

    img_path = os.path.join(datadir,cube_down )
    img_2 = cv2.imread(img_path)
    for img in [img_1, img_2]:
        box, _ = cube_seg.detect_cube(img)
        mid_points = cube_seg.get_midpoints(img, box)
        rvec, tvec,_, inliners= estimate_cube_pose(mid_points, K, dist_coeffs)
        success, proj2d_s = get_surfaces_Q1_Q2_Q3(rvec=rvec, tvec=tvec, K=K, dist_coeffs=dist_coeffs, inliners=inliners)
        solver.update_cube(img, proj2d_s)

    cube_string = solver.cube_state.get_cube_string_notation(altenative=True)
    # https://twinone.github.io/rubik-web/
    c2c = {'Y':'U', 'R':'F', 'B':'L', 'O':'B', 'G':'R', 'W':'D'}
    visual_cube_notation = ''.join([c2c[s] for s in cube_string])
    assert visual_cube_notation == 'URRLRRBDDBBLDDLLDDULRUBURDDFFUFUBLUDFURRFRBLFBBFFLBUFL'

    solver_notation = solver.cube_state.get_kociemba_string_notation()
    solution = kociemba.solve(solver_notation)
    assert solution == "U2 B2 U2 D' R' L' D L' U2 B' R F2 U' F2 R2 U' F2 R2 D B2 U"

    if plot:
        solver.cube_state.plot(img_1)
        cv2.imshow("Upper Cube", img_1)
        cv2.imshow("Lower Cube", img_2)
        cv2.putText(img_1, "Upper", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img_2, "Lower", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

