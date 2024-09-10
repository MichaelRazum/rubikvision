import kociemba
import os

import pytest
import numpy as np
import cv2

from rubikvision.color_classifier import ColorClassiferKmeans
from rubikvision.cube_pose import  estimate_cube_pose, get_surfaces_Q1_Q2_Q3
from rubikvision.cube_solver import CubeSolver, find_corners, find_closest_corners, align_cube_surfaces, CubeState, \
    CubeStateColorError, expand_solution, STATE_SOLVED


@pytest.fixture(scope='function')
def solver():
    clf = ColorClassiferKmeans().load('color_classifier_test.pkl')
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

    assert cube_state.current_upper == 'red'
    assert cube_state.current_left == 'orange'
    assert cube_state.current_front == 'green'

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
    assert cube_state.current_upper == 'white'
    assert cube_state.current_left == 'blue'
    assert cube_state.current_front == 'yellow'

    assert cube_state.down == down
    assert cube_state.right == front
    assert cube_state.back  == left

def test_check_orientation_consitency():
    cube_state = CubeState()
    upper= ['blue' for _ in range(9)]
    left  = ['white' for _ in range(9)]
    front = ['red' for _ in range(9)]
    cube_state.update(upper=upper, left=left, front=front)

    assert cube_state._check_orientation_consitency(mid_upper='blue', mid_left='green', mid_front='white') == True


    down = ['green' for _ in range(9)]
    left = ['orange' for _ in range(9)]
    front = ['yellow' for _ in range(9)]
    cube_state.update(upper=down, left=left, front=front)

    assert cube_state._check_orientation_consitency(mid_upper='blue', mid_left='white', mid_front='red') == True
    assert cube_state._check_orientation_consitency(mid_upper='white', mid_left='blue', mid_front='red') == False

    assert cube_state._check_orientation_consitency(mid_upper='yellow', mid_left='blue', mid_front='red') == True
    assert cube_state._check_orientation_consitency(mid_upper='yellow', mid_left='red', mid_front='blue') == False

    assert cube_state._check_orientation_consitency(mid_upper='blue', mid_left='green', mid_front='white') == False
    assert cube_state._check_orientation_consitency(mid_upper='green', mid_left='blue', mid_front='white') == False

def test_from_state():
    cube_state = CubeState()
    upper= ['U' for _ in range(9)]
    left  = ['L' for _ in range(9)]
    front = ['F' for _ in range(9)]
    cube_state.update(upper=upper, left=left, front=front)
    assert cube_state.current_upper == 'U'
    assert cube_state.current_left  == 'L'
    assert cube_state.current_front == 'F'

    cube_state_new = cube_state.copy_from_state(current_upper='D',
                                                current_front='R',
                                                current_left='B',)
    assert cube_state_new.current_upper == 'D'
    assert cube_state_new.current_left  == 'B'
    assert cube_state_new.current_front == 'R'




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

def test_solution_and_next_state():
    visualize = False
    state_1 = CubeState(current_upper='white', current_left='green', current_front='red',
                     upper=['white', 'yellow', 'white', 'orange', 'yellow', 'blue', 'white', 'yellow', 'yellow'],
                     left=['orange', 'green', 'orange', 'red', 'orange', 'blue', 'blue', 'red', 'red'],
                     front=['green', 'red', 'green', 'white', 'blue', 'green', 'yellow', 'blue', 'orange'],
                     down=['green', 'orange', 'blue', 'blue', 'white', 'red', 'yellow', 'white', 'red'],
                     right=['orange', 'yellow', 'red', 'yellow', 'red', 'white', 'yellow', 'green', 'green'],
                     back=['blue', 'orange', 'blue', 'orange', 'green', 'white', 'white', 'green', 'red'])
    while True:
        string_1 = state_1.get_kociemba_string_notation()
        solution_1 = kociemba.solve(string_1)
        solution_1 = expand_solution(solution_1)
        print(solution_1)
        if solution_1 == "":
            break
        state_2 = state_1.get_next_cube(solution_1)

        if visualize:
            img_1 = np.zeros((420, 420, 3))
            img_2 = np.zeros((420, 420, 3))
            # https://twinone.github.io/rubik-web/
            cube_string=state_1.get_cube_string_notation(altenative=True)
            c2c = {'Y':'U', 'R':'F', 'B':'L', 'O':'B', 'G':'R', 'W':'D'}
            cube_string_1 = ''.join([c2c[s] for s in cube_string])
            cube_string=state_2.get_cube_string_notation(altenative=True)
            cube_string_2 = ''.join([c2c[s] for s in cube_string])
            print(cube_string_1)
            print(cube_string_2)

            state_1.plot(img_1)
            state_2.plot(img_2)

            cv2.imshow("original", img_1)
            cv2.imshow("second", img_2)
            cv2.waitKey(0)
            cv2.waitKey(0)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        string_2 = state_2.get_kociemba_string_notation()
        solution_2 = kociemba.solve(string_2)
        solution_2 = expand_solution(solution_2)

        assert solution_1.split(' ')[1:] in [[], solution_2.split(' ') ]

        #check Reverse
        action = solution_1.split(' ')[0]
        action = action + "'" if len(action) == 1 else action[0]
        state_1_recovered = state_2.get_next_cube(action)
        assert state_1 == state_1_recovered

        # next iteration
        state_1 = state_2

        if string_2 == STATE_SOLVED:
            break

def test_detect_rotation():
    green_ordered = ['green_1', 'green_2', 'green_3',
                     'green_4', 'green', 'green_5',
                     'green_6', 'green_7', 'green_8'
                     ]
    cube = CubeState(current_upper='white', current_left='green', current_front='red',
                    upper=green_ordered,
                    left=['red']*9,
                    front=['white']*9,
                    down=['blue']*9,
                    right=['yellow']*9,
                    back=['orange']*9)

    cube_map_0 = dict(upper=green_ordered,
                    left=['red']*9,
                    front=['white']*9)
    cube_map_1 = dict(upper=green_ordered,
                    left=['red']*9,
                    front=['white']*9)
    assert cube.detect_rotation(cube_map_0, cube_map_1, ERROR_TOL=0) is None

    cube_map_0 = dict(upper=green_ordered,
                     left=['red' for _ in range(9)],
                     front=['white' for _ in range(9)])
    cube_map_1 = dict(upper=['green_3', 'green_5', 'green_8',
                             'green_2', 'green',  'green_7',
                             'green_1', 'green_4', 'green_6'
                             ],
                      left=['orange']*3 + ['red']*6,
                      front=['red']*3 + ['white']*6)
    cube_target = cube.get_next_cube("U'")
    assert cube.detect_rotation(cube_map_0, cube_map_1,ERROR_TOL=0) == "U'"
    assert cube_target.left == cube_map_1['left']
    assert cube_target.front == cube_map_1['front']
    assert cube_target.upper == cube_map_1['upper']
    # detect with one error
    cube_map_1['front'][0] = 'orange'
    assert cube.detect_rotation(cube_map_0, cube_map_1, ERROR_TOL=0) is None
    assert cube.detect_rotation(cube_map_0, cube_map_1, ERROR_TOL=1) == "U'"


    cube_map_0 = dict(upper=green_ordered,
                     left=['red' for _ in range(9)],
                     front=['white' for _ in range(9)])
    cube_map_1 = dict(upper=['green_6', 'green_4', 'green_1',
                             'green_7', 'green',  'green_2',
                             'green_8', 'green_5', 'green_3'
                             ],
                      left=['white']*3 + ['red']*6,
                      front=['yellow']*3 + ['white']*6)
    cube_target = cube.get_next_cube("U")
    assert cube.detect_rotation(cube_map_0, cube_map_1) == "U"
    assert cube_target.left == cube_map_1['left']
    assert cube_target.front == cube_map_1['front']

    cube_map_1['left'][0] = 'orange'
    assert cube.detect_rotation(cube_map_0, cube_map_1, ERROR_TOL=0) is None
    assert cube.detect_rotation(cube_map_0, cube_map_1, ERROR_TOL=1) == "U"

def test_cube_rotation_prod_error():
    cube = CubeState(current_upper='white', current_left='red', current_front='blue',
              upper=['white', 'yellow', 'white', 'orange', 'white', 'green', 'orange', 'white', 'blue'],
              left=['orange', 'yellow', 'blue', 'yellow', 'green', 'white', 'yellow', 'green', 'yellow'],
              front=['yellow', 'green', 'white', 'blue', 'red', 'red', 'orange', 'white', 'red'],
              down=['green', 'red', 'yellow', 'orange', 'yellow', 'yellow', 'green', 'blue', 'white'],
              right=['red', 'red', 'green', 'blue', 'blue', 'orange', 'blue', 'red', 'green'],
              back=['red', 'blue', 'blue', 'white', 'orange', 'green', 'orange', 'orange', 'red'])

    old_state = {'upper': ['orange', 'orange', 'white', 'white', 'white', 'yellow', 'blue', 'green', 'white'],
                 'left': ['yellow', 'green', 'white', 'blue', 'red', 'red', 'orange', 'white', 'red'],
                 'front': ['red', 'red', 'green', 'blue', 'blue', 'red', 'blue', 'red', 'white']}
    new_state = {'upper': ['white', 'yellow', 'white', 'orange', 'white', 'green', 'orange', 'white', 'blue'],
                 'left': ['orange', 'yellow', 'blue', 'blue', 'red', 'red', 'orange', 'white', 'red'],
                 'front': ['yellow', 'green', 'white', 'blue', 'blue', 'red', 'blue', 'red', 'green']}
    rot = cube.detect_rotation(old_state, new_state, ERROR_TOL=1)
    assert rot is not None
    # Case no cube slice rotation, only rotation of the cube
    cube_state = CubeState(current_upper='yellow', current_left='green', current_front='orange',
                           upper=['blue', 'white', 'orange', 'green', 'white', 'orange', 'white', 'yellow', 'yellow'],
                           left=['red', 'red', 'orange', 'yellow', 'green', 'blue', 'orange', 'white', 'orange'],
                           front=['green', 'red', 'blue', 'red', 'red', 'blue', 'blue', 'blue', 'red'],
                           down=['white', 'yellow', 'white', 'red', 'yellow', 'blue', 'green', 'orange', 'green'],
                           right=['red', 'yellow', 'blue', 'white', 'blue', 'orange', 'green', 'orange', 'red'],
                           back=['yellow', 'green', 'white', 'white', 'orange', 'green', 'yellow', 'green', 'yellow'])
    old_state = {'upper': ['green', 'red', 'white', 'orange', 'yellow', 'yellow', 'green', 'blue', 'white'],
                 'left': ['yellow', 'green', 'yellow', 'green', 'orange', 'white', 'white', 'green', 'yellow'],
                 'front': ['red', 'orange', 'green', 'orange', 'blue', 'white', 'blue', 'yellow', 'white']}
    new_state = {'upper': ['white', 'yellow', 'white', 'red', 'yellow', 'blue', 'green', 'orange', 'green'],
                 'left': ['orange', 'white', 'orange', 'blue', 'green', 'yellow', 'orange', 'red', 'red'],
                         'front': ['yellow', 'green', 'yellow', 'green', 'orange', 'white', 'white', 'green', 'yellow']}
    rot = cube.detect_rotation(old_state, new_state, ERROR_TOL=1)
    assert rot is None


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

