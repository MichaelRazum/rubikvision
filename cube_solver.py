from collections import Counter
from dataclasses import dataclass, field

from typing import List

import cv2
import webcolors

from color_classifier import ColorClassifer
import numpy as np

from cube_pose import rotate_90_clockwise, list_to_grid, grid_to_list, reverse_rows

def _flip_list(list, idx):
    assert len(list) == 9
    if idx in {0, 8}:
        return [list[0], list[3], list[6],
                list[1], list[4], list[7],
                list[2], list[5], list[8]]

    elif idx in {2,6}:
        return [list[8], list[5], list[2],
                list[7], list[4], list[1],
                list[6], list[3], list[0]]
    elif idx in {1,7}:
        return [list[2],list[1], list[0],
                list[5],list[4], list[3],
                list[8],list[7], list[6],]
    elif idx in {3,5}:
        return [list[6], list[7], list[8],
                list[3], list[4], list[5],
                list[0], list[1], list[2],]


    else:
        raise Exception('not implemented')

def find_corners(surface):
    return [
        surface[0],  # Top-left
        surface[2],  # Top-right
        surface[-3], # Bottom-left
        surface[-1]  # Bottom-right
    ]


def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def find_closest_corners(corners):
    min_max_dist = float('inf')
    closest_triplet = None
    for i in range(4):
        for j in range(4):
            for k in range(4):
                c1, c2, c3 = corners[0][i], corners[1][j], corners[2][k]
                max_dist = max(distance(c1, c2), distance(c2, c3), distance(c3, c1))
                if max_dist < min_max_dist:
                    min_max_dist = max_dist
                    closest_triplet = (c1, c2, c3)
    return closest_triplet


def get_top_left_right_surfaces(proj_2d_s):
    proj_2d_s = [np.array(surface) for surface in proj_2d_s]
    mean_y_coords = [np.mean(surface[:, 1]) for surface in proj_2d_s]
    top_index = np.argmin(mean_y_coords)

    non_top_indices = [i for i in range(3) if i != top_index]
    mean_x_coords = [np.mean(proj_2d_s[i][:, 0]) for i in non_top_indices]
    left_index = non_top_indices[np.argmin(mean_x_coords)]
    right_index = non_top_indices[np.argmax(mean_x_coords)]
    top = proj_2d_s[top_index]
    left = proj_2d_s[left_index]
    right = proj_2d_s[right_index]
    return top, left, right


def __rotate_surface(surf_in, corner_idx, target_idx):
    grid = list_to_grid(surf_in.tolist())
    for _ in range(4):
        surf = grid_to_list(grid)
        if list(surf_in[corner_idx]) == list(surf[target_idx]):
            return surf
        else:
            grid = rotate_90_clockwise(grid)
    raise Exception('could not rotate to target')




def __align_cube_direction(surf_1, idx_1, corner_idx_1, surf_2, idx_2, corner_idx_2):
    dist = float('inf')
    flip_1_final, flip_2_final = False, False
    for flip_1 in [False, True]:
        for flip_2 in [False, True]:
            s1 = _flip_list(surf_1, corner_idx_1)[idx_1] if flip_1 else surf_1[idx_1]
            s2 = _flip_list(surf_2, corner_idx_2)[idx_2] if flip_2 else surf_2[idx_2]
            if distance(s1, s2) < dist:
                dist = distance(s1, s2)
                flip_1_final = flip_1
                flip_2_final = flip_2
    surf_1_ret = _flip_list(surf_1, corner_idx_1) if flip_1_final else surf_1
    surf_2_ret = _flip_list(surf_2, corner_idx_2) if flip_2_final else surf_2
    return surf_1_ret, surf_2_ret




class CubeStateColorError(Exception):
    pass

@dataclass
class CubeState:
    """
                     -----
                  | U1 U2 U3 |
                  | U4 U5 U6 |
                  | U7 U8 U9 |
        -----   -----   -----   -----
      | L1 L2 L3 | F1 F2 F3 | R1 R2 R3 | B1 B2 B3 |
      | L4 L5 L6 | F4 F5 F6 | R4 R5 R6 | B4 B5 B6 |
      | L7 L8 L9 | F7 F8 F9 | R7 R8 R9 | B7 B8 B9 |
        -----   -----   -----   -----
                | D1 D2 D3 |
                | D4 D5 D6 |
                | D7 D8 D9 |
                    -----
    """
    upper: List = field(default_factory=list)
    left: List = field(default_factory=list)
    front: List = field(default_factory=list)
    down: List = field(default_factory=list)
    right: List = field(default_factory=list)
    back:  List= field(default_factory=list)

    def check_consistency(self, top, left, right):
        if len(set(top+left+right)) > 6:
            raise CubeStateColorError('number of unique colors > 6')
        if len({top[4], left[4], right[4]}) != 3:
            raise CubeStateColorError('Midpoint color should be unique')
        if max(Counter(top+left+right).values()) > 9:
            raise CubeStateColorError('Found more than 9 colors')

    def update(self, upper, left, front):
        self.check_consistency(upper, left, front)
        if self.upper == []:
            self.upper = upper
            self.left = left
            self.front = front
        else:
            mid_colors = {self.upper[4], self.left[4], self.front[4]}
            mid_colors_new = {upper[4], left[4], front[4]}
            if mid_colors & mid_colors_new :
                raise CubeStateColorError(f'unique squares needed {mid_colors & mid_colors_new}')
            self.down =  _flip_list(_flip_list(upper, 1),0)
            self.right = _flip_list(_flip_list(front,2),0)
            self.back = _flip_list(_flip_list(left,2),0)

    def get_cube_string_notation(self, altenative =False):
        # UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB
        if altenative is False:
            ordered_surface = [self.upper,
                               self.right,
                               self.front,
                               self.down,
                               self.left,
                               self.back]
        else:
            # UUUUUUUUULLLLLLLLLFFFFFFFFFRRRRRRRRRBBBBBBBBBDDDDDDDDD
            ordered_surface = [self.upper,
                               self.left,
                               self.front,
                               self.right,
                               self.back,
                               self.down]
        string_notation = ''
        for surf in ordered_surface:
            string_notation += "".join([ch[0].upper() for ch in surf])
        return string_notation

    def _plot_get_color(self, color_name, color_format='bgr'):
        try:
            color = webcolors.name_to_rgb(color_name)
        except ValueError:
            color = (128, 128, 128)  # Return grey for unknown colors

        if color_format == 'bgr':
            return (color.blue, color.green, color.red)
        elif color_format == 'rgb':
            return (color.red, color.green, color.blue)
        else:
            raise ValueError("Unsupported color format. Use 'rgb' or 'bgr'.")

    def plot(self, image):
        cube_size = 45
        start_x = 0
        start_y = 0
        square_size = cube_size // 3

        def draw_face(face, start_x, start_y):
            for i in range(9):
                x = start_x + (i % 3) * square_size
                y = start_y + (i // 3) * square_size
                if i < len(face) and face[i]:
                    color = self._plot_get_color(face[i])
                else:
                    color = (128, 128, 128)  # Grey for unknown or empty
                cv2.rectangle(image, (x, y), (x + square_size, y + square_size), color, -1)
                cv2.rectangle(image, (x, y), (x + square_size, y + square_size), (0, 0, 0), 1)

            border_thickness = 2
            cv2.rectangle(image,
                          (start_x, start_y),
                          (start_x + 3 * square_size, start_y + 3 * square_size),
                          (0, 0, 0), border_thickness)

        # Draw faces
        faces = [
            (self.upper, start_x + cube_size, start_y),
            (self.left, start_x, start_y + cube_size),
            (self.front, start_x + cube_size, start_y + cube_size),
            (self.right, start_x + cube_size * 2, start_y + cube_size),
            (self.back, start_x + 3 * cube_size, start_y + cube_size),
            (self.down, start_x + cube_size, start_y + cube_size * 2)
        ]

        for face, x, y in faces:
            draw_face(face, x, y)




def align_cube_surfaces(proj2d_s, is_back=False):
    """
                -----
                  | U1 U2 U3 |
                  | U4 U5 U6 |
                  | U7 U8 U9 |
        -----   -----   -----   -----
      | L1 L2 L3 | F1 F2 F3 | R1 R2 R3 | B1 B2 B3 |
      | L4 L5 L6 | F4 F5 F6 | R4 R5 R6 | B4 B5 B6 |
      | L7 L8 L9 | F7 F8 F9 | R7 R8 R9 | B7 B8 B9 |
        -----   -----   -----   -----
                 | D1 D2 D3 |
                 | D4 D5 D6 |
                 | D7 D8 D9 |
                   -----
    """
    if is_back:
        edg_idx_top = 2
        edg_idx_left = 0
        edg_idx_right = 8
        top_right_idxs = [0, 6]
        top_left_idx = [8, 6]
    else:
        edg_idx_top = 6
        edg_idx_left = 2
        edg_idx_right = 0
        top_right_idxs = [8, 2]
        top_left_idx = [0, 0]

    top, left, right = get_top_left_right_surfaces(proj2d_s)
    all_corners = [find_corners(surface) for surface in [top, left, right]]
    top_corner, left_corner, right_corner = find_closest_corners(all_corners)

    # align surfaces F7 L9 D1 for Front = True
    top_idx = np.where((top == top_corner).all(axis=1))[0][0]
    left_idx = np.where((left == left_corner).all(axis=1))[0][0]
    right_idx = np.where((right == right_corner).all(axis=1))[0][0]
    top_aligned = __rotate_surface(top, corner_idx=top_idx, target_idx=edg_idx_top)
    left_aligned = __rotate_surface(left, corner_idx=left_idx, target_idx=edg_idx_left)
    right_aligned = __rotate_surface(right, corner_idx=right_idx, target_idx=edg_idx_right)

    # align F9 and D3
    top_rotated, right_rotated = __align_cube_direction(surf_1=top_aligned,
                                                        idx_1=top_right_idxs[0],
                                                        corner_idx_1=edg_idx_top,
                                                        surf_2=right_aligned,
                                                        idx_2=top_right_idxs[1],
                                                        corner_idx_2=edg_idx_right)
    # align F1 and L3
    _, left_rotated = __align_cube_direction(surf_1=top_aligned,
                                             idx_1=top_left_idx[0],
                                             corner_idx_1=edg_idx_top,
                                             surf_2=left_aligned,
                                             corner_idx_2=edg_idx_left,
                                             idx_2=top_left_idx[1])
    return {
        'upper':  top_rotated,
        'left':  left_rotated,
        'front': right_rotated
    }




class CubeSolver:
    def __init__(self, clf: ColorClassifer):
        self.clf = clf
        self.cube_state = CubeState()

    def is_estimation_ok(self, inliners, MIN_INLINERS=12):
        return len(inliners) >= MIN_INLINERS

    def estimate_color(self, img, point):
        return self.clf.predict(img[int(point[1]), int(point[0]),])

    def map_surface(self, img, surf_mids):
        colors = []
        for p in surf_mids:
            c = self.estimate_color(img, p)
            colors.append(c)
        return colors

    def map_cube(self, img, surfaces):
        img = self.clf.bgr2clf_format(img)
        colors = dict()
        for name, surf in surfaces.items():
            surf_colors = self.map_surface(img, surf)
            colors[name] = surf_colors
        return colors

    def update_cube(self, img, proj2d_s):
        alligned_surfaces = align_cube_surfaces(proj2d_s)
        color_map = self.map_cube(img, alligned_surfaces)
        self.cube_state.update(upper=color_map['upper'], left=color_map['left'], front=color_map['front'])