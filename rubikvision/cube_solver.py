import enum
import threading
import traceback
from collections import Counter, deque
from copy import deepcopy
from dataclasses import dataclass, field, replace

from typing import List

import cv2
import kociemba
import torch
import webcolors
import numpy as np
from webcolors import IntegerRGB

from rubikvision.color_classifier import ColorClassifer, ColorClassiferKmeans
from rubikvision.cube_detection import CubeSegmentation, draw_cube_adaptive_edges, draw_bounding_box, highlight_points
from rubikvision.cube_pose import rotate_90_clockwise, list_to_grid, grid_to_list, reverse_rows, get_cube_edges, \
    get_surfaces_Q1_Q2_Q3, estimate_cube_pose

STATE_SOLVED = 'UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB'

def expand_solution(sol):
    return ' '.join([ch if '2' not in ch else ch[0] + ' ' + ch[0] for ch in sol.split(' ')])

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
    current_upper: str = '' #U5
    current_left: str = ''  #L5
    current_front: str = '' #F5

    upper: List = field(default_factory=list)
    left: List = field(default_factory=list)
    front: List = field(default_factory=list)
    down: List = field(default_factory=list)
    right: List = field(default_factory=list)
    back:  List= field(default_factory=list)

    @property
    def side2midpoint(self):
        return {'U': self.upper[4] if self.upper else None,
                'L': self.left[4] if self.left else None,
                'F': self.front[4] if self.front else None,
                'B': self.back[4] if self.back else None,
                'R': self.right[4] if self.right else None,
                'D': self.down[4] if self.down else None}
    @property
    def midpoint2side(self):
        side2midpoint = self.side2midpoint
        return {v: k for k, v in side2midpoint.items()}


    def __eq__(self, other):
        if not isinstance(other, CubeState):
            return NotImplemented

        if (self.upper == other.upper and
            self.left == other.left and
            self.front == other.front and
            self.down == other.down and
            self.right == other.right and
            self.back == other.back) is False:
            return False
        orientation_1 = [self.current_upper, self.current_left, self.current_front]
        orientation_2 = [other.current_upper, other.current_left, other.current_front]

        for or1, or2 in zip(orientation_1, orientation_2):
            if or1!=or2:
                if or1 != "" and or1[0] in {'!', '*'}:
                    continue
                if or2 != "" and or2[0] in {'!', '*'}:
                    continue
                return False

        not_allowed_colors = {o[1:] for o in orientation_1 if o!="" and o[0]=="!"}
        if not_allowed_colors & set(orientation_2):
            return False
        not_allowed_colors = {o[1:] for o in orientation_2 if o!="" and o[0]=="!"}
        if not_allowed_colors & set(orientation_1):
            return False
        return True

    def copy_from_state(self, **changes):
        new_state = deepcopy(self)
        return replace(new_state, **changes)

    def get_next_cube(self, solution):
        if solution == "":
            return self.copy_from_state()
        move = solution.split(' ')[0]
        cube_ret = self.copy_from_state()
        if move[0] == 'F':
            if "'" in move:
                cube_ret.front = grid_to_list(rotate_90_clockwise(list_to_grid(self.front)))[::-1]
                cube_ret.right[6], cube_ret.right[3], cube_ret.right[0] = self.down[0],  self.down[1],  self.down[2]
                cube_ret.down[0],  cube_ret.down[1],  cube_ret.down[2] =  self.left[2],  self.left[5],  self.left[8]
                cube_ret.left[8],  cube_ret.left[5],  cube_ret.left[2] =  self.upper[6], self.upper[7], self.upper[8]
                cube_ret.upper[6], cube_ret.upper[7], cube_ret.upper[8] = self.right[0], self.right[3], self.right[6]
            else:
                cube_ret.front = grid_to_list(rotate_90_clockwise(list_to_grid(self.front)))
                cube_ret.down[0], cube_ret.down[1], cube_ret.down[2] = self.right[6], self.right[3], self.right[0]
                cube_ret.left[2], cube_ret.left[5], cube_ret.left[8] = self.down[0], self.down[1], self.down[2]
                cube_ret.upper[6], cube_ret.upper[7], cube_ret.upper[8] = self.left[8], self.left[5], self.left[2]
                cube_ret.right[0], cube_ret.right[3], cube_ret.right[6] = self.upper[6], self.upper[7], self.upper[8]
            return cube_ret
        if move[0] == 'B':
            if "'" in move:
                cube_ret.back = grid_to_list(rotate_90_clockwise(list_to_grid(self.back)))[::-1]
                cube_ret.right[2], cube_ret.right[5], cube_ret.right[8] = self.upper[0], self.upper[1], self.upper[2]
                cube_ret.down[8],  cube_ret.down[7],  cube_ret.down[6]  = self.right[2], self.right[5], self.right[8]
                cube_ret.left[0],  cube_ret.left[3],  cube_ret.left[6]  = self.down[6],  self.down[7],  self.down[8]
                cube_ret.upper[2], cube_ret.upper[1], cube_ret.upper[0] = self.left[0],  self.left[3],  self.left[6]
            else:
                cube_ret.back = grid_to_list(rotate_90_clockwise(list_to_grid(self.back)))
                cube_ret.upper[0], cube_ret.upper[1], cube_ret.upper[2] = self.right[2], self.right[5], self.right[8]
                cube_ret.right[2], cube_ret.right[5], cube_ret.right[8] = self.down[8], self.down[7], self.down[6]
                cube_ret.down[6], cube_ret.down[7], cube_ret.down[8] = self.left[0], self.left[3], self.left[6]
                cube_ret.left[0], cube_ret.left[3], cube_ret.left[6] = self.upper[2], self.upper[1], self.upper[0]
            return cube_ret
        if move[0] == 'U':
            if "'" in move:
                cube_ret.upper = grid_to_list(rotate_90_clockwise(list_to_grid(self.upper)))[::-1]
                cube_ret.right[0], cube_ret.right[1], cube_ret.right[2] = self.front[0], self.front[1], self.front[2]
                cube_ret.back[0],  cube_ret.back[1],  cube_ret.back[2] =  self.right[0], self.right[1], self.right[2]
                cube_ret.left[0],  cube_ret.left[1],  cube_ret.left[2] =  self.back[0],  self.back[1],  self.back[2]
                cube_ret.front[0], cube_ret.front[1], cube_ret.front[2] = self.left[0],  self.left[1],  self.left[2]
            else:
                cube_ret.upper = grid_to_list(rotate_90_clockwise(list_to_grid(self.upper)))
                cube_ret.front[0], cube_ret.front[1], cube_ret.front[2] = self.right[0], self.right[1], self.right[2]
                cube_ret.right[0], cube_ret.right[1], cube_ret.right[2] = self.back[0], self.back[1], self.back[2]
                cube_ret.back[0], cube_ret.back[1], cube_ret.back[2] = self.left[0], self.left[1], self.left[2]
                cube_ret.left[0], cube_ret.left[1], cube_ret.left[2] = self.front[0], self.front[1], self.front[2]
            return cube_ret
        if move[0] == 'D':
            if "'" in move:
                cube_ret.down = grid_to_list(rotate_90_clockwise(list_to_grid(self.down)))[::-1]
                cube_ret.left[6],  cube_ret.left[7],  cube_ret.left[8] =  self.front[6], self.front[7], self.front[8]
                cube_ret.front[6], cube_ret.front[7], cube_ret.front[8] = self.right[6], self.right[7], self.right[8]
                cube_ret.right[6], cube_ret.right[7], cube_ret.right[8] = self.back[6],  self.back[7],  self.back[8]
                cube_ret.back[6],  cube_ret.back[7],  cube_ret.back[8] =  self.left[6],  self.left[7],  self.left[8]
            else:
                cube_ret.down = grid_to_list(rotate_90_clockwise(list_to_grid(self.down)))
                cube_ret.front[6], cube_ret.front[7], cube_ret.front[8] = self.left[6], self.left[7], self.left[8]
                cube_ret.right[6], cube_ret.right[7], cube_ret.right[8] = self.front[6], self.front[7], self.front[8]
                cube_ret.back[6], cube_ret.back[7], cube_ret.back[8] = self.right[6], self.right[7], self.right[8]
                cube_ret.left[6], cube_ret.left[7], cube_ret.left[8]= self.back[6], self.back[7], self.back[8]
            return cube_ret
        if move[0] == 'L':
            if "'" in move:
                cube_ret.left = grid_to_list(rotate_90_clockwise(list_to_grid(self.left)))[::-1]
                cube_ret.back[8],  cube_ret.back[5],  cube_ret.back[2] =  self.upper[0], self.upper[3], self.upper[6]
                cube_ret.down[6],  cube_ret.down[3],  cube_ret.down[0] =  self.back[2],  self.back[5],  self.back[8]
                cube_ret.front[0], cube_ret.front[3], cube_ret.front[6] = self.down[0],  self.down[3],  self.down[6]
                cube_ret.upper[0], cube_ret.upper[3], cube_ret.upper[6] = self.front[0], self.front[3], self.front[6]
            else:
                cube_ret.left = grid_to_list(rotate_90_clockwise(list_to_grid(self.left)))
                cube_ret.upper[0], cube_ret.upper[3], cube_ret.upper[6] = self.back[8], self.back[5], self.back[2]
                cube_ret.back[2], cube_ret.back[5], cube_ret.back[8] = self.down[6], self.down[3], self.down[0]
                cube_ret.down[0], cube_ret.down[3], cube_ret.down[6] = self.front[0], self.front[3], self.front[6]
                cube_ret.front[0], cube_ret.front[3], cube_ret.front[6] = self.upper[0], self.upper[3], self.upper[6]
            return cube_ret
        if move[0] == 'R':
            if "'" in move:
                cube_ret.right = grid_to_list(rotate_90_clockwise(list_to_grid(self.right)))[::-1]
                cube_ret.down[2],  cube_ret.down[5],  cube_ret.down[8] =  self.front[2], self.front[5],self.front[8]
                cube_ret.back[6],  cube_ret.back[3],  cube_ret.back[0] =  self.down[2],  self.down[5], self.down[8]
                cube_ret.upper[8], cube_ret.upper[5], cube_ret.upper[2] = self.back[0],  self.back[3], self.back[6]
                cube_ret.front[2], cube_ret.front[5], cube_ret.front[8] = self.upper[2], self.upper[5],self.upper[8]
            else:
                cube_ret.right = grid_to_list(rotate_90_clockwise(list_to_grid(self.right)))
                cube_ret.front[2], cube_ret.front[5], cube_ret.front[8] = self.down[2], self.down[5], self.down[8]
                cube_ret.down[2], cube_ret.down[5], cube_ret.down[8] = self.back[6], self.back[3], self.back[0]
                cube_ret.back[0], cube_ret.back[3], cube_ret.back[6] = self.upper[8], self.upper[5], self.upper[2]
                cube_ret.upper[2], cube_ret.upper[5], cube_ret.upper[8] = self.front[2], self.front[5], self.front[8]
            return cube_ret

    def check_consistency(self, top, left, right):
        if len(set(top+left+right)) > 6:
            raise CubeStateColorError('number of unique colors > 6')
        if len({top[4], left[4], right[4]}) != 3:
            raise CubeStateColorError('Midpoint color should be unique')
        if max(Counter(top+left+right).values()) > 9:
            raise CubeStateColorError('Found more than 9 colors')

    def check_consistency_full_map(self):
        cnt = Counter(self.upper+
                      self.left+
                      self.front+
                      self.down+
                      self.right+
                      self.back)
        state_ok = max(cnt.values()) == min(cnt.values()) == 9
        if not state_ok:
            print(f'got bad state {cnt}')
        return state_ok

    def _check_orientation_consitency(self,mid_upper, mid_left, mid_front):
        if self.down == []  and self.upper== []:
            return True
        if self.down == [] and self.upper != []:
            if mid_upper != self.upper[4] and mid_left == self.left[4] and mid_front == self.front[4]:
                return False
            if mid_upper == self.upper[4] and mid_left != self.left[4] and mid_front == self.front[4]:
                return False
            if mid_upper == self.upper[4] and mid_left == self.left[4] and mid_front != self.front[4]:
                return False
            return True
        m_up, m_left, m_front = self.upper[4], self.left[4], self.front[4]
        m_back, m_right, m_down= self.back[4], self.right[4], self.down[4]
        tripples = [
            (m_up, m_left, m_front),
            (m_up, m_front, m_right),
            (m_up, m_right, m_back),
            (m_up, m_back, m_left),
            (m_down, m_back, m_right),
            (m_down, m_right, m_front),
            (m_down, m_front, m_left),
            (m_down, m_left, m_back),
        ]

        tripples_comb = set()
        for tripple in tripples:
            tripples_comb.add(tripple)
            tripples_comb.add((tripple[1], tripple[2], tripple[0]))
            tripples_comb.add((tripple[2], tripple[0], tripple[1]))

        if (mid_upper, mid_left, mid_front) in tripples_comb:
            return True
        else:
            return False


    def update(self, upper, left, front):
        self.check_consistency(upper, left, front)
        if self._check_orientation_consitency(upper[4], left[4], front[4]):
            self.current_upper = upper[4]
            self.current_left  = left[4]
            self.current_front = front[4]

        if self.upper == []:
            self.upper = upper
            self.left = left
            self.front = front
        elif self.down == []:
            mid_colors = {self.upper[4], self.left[4], self.front[4]}
            mid_colors_new = {upper[4], left[4], front[4]}
            if mid_colors & mid_colors_new :
                raise CubeStateColorError(f'unique squares needed {mid_colors & mid_colors_new}')
            self.down =  _flip_list(_flip_list(upper, 1),0)
            self.right = _flip_list(_flip_list(front,2),0)
            self.back = _flip_list(_flip_list(left,2),0)

    def reset(self, part='down'):
        self.down = []
        self.right = []
        self.back = []
        if part == 'all':
            self.upper = []
            self.left = []
            self.front = []

    def detect_rotation(self, last_color_map, color_map, ERROR_TOL=1):
        mid_point_0 = last_color_map['upper'][4]
        mid_point_1 = color_map['upper'][4]
        if mid_point_0 != mid_point_1:
            return

        upper_last = np.array(last_color_map['upper'])
        upper_current = np.array(color_map['upper'])
        left_last = np.array(last_color_map['left'][:3])
        left_current = np.array(color_map['left'][:3])
        front_last = np.array(last_color_map['front'][:3])
        front_current = np.array(color_map['front'][:3])

        left_last_lower = np.array(last_color_map['left'][3:])
        left_current_lower = np.array(color_map['left'][3:])
        front_last_lower = np.array(last_color_map['front'][3:])
        front_current_lower = np.array(color_map['front'][3:])

        rotation_counter_clock = upper_last[[2, 5, 8, 1, 4, 7, 0, 3, 6]]
        rotation_clock = upper_last[[6, 3, 0, 7, 4, 1, 8, 5, 2]]


        error_unefected_rows = np.sum(left_last_lower != left_current_lower) + np.sum(front_last_lower != front_current_lower)
        ccw_errors = np.sum(left_last != front_current) + np.sum(upper_current != rotation_counter_clock)  +error_unefected_rows
        cw_errors = np.sum(front_last != left_current) + np.sum(upper_current != rotation_clock) + error_unefected_rows

        side = self.midpoint2side[mid_point_0]
        if ccw_errors <= ERROR_TOL:
            return side + "'"
        elif cw_errors <= ERROR_TOL:
            return side

        return None  # No rotation detected within error tolerance
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

    def get_kociemba_string_notation(self):
        mapping = {self.upper[4]: 'U',
                   self.right[4]: 'R',
                   self.front[4]: 'F',
                   self.down[4]: 'D',
                   self.left[4]: 'L',
                   self.back[4]: 'B'}
        ordered_surface = [self.upper,
                           self.right,
                           self.front,
                           self.down,
                           self.left,
                           self.back]
        string_notation = ''
        for surf in ordered_surface:
            string_notation += "".join([mapping[str] for str in surf])
        return string_notation


    def _plot_get_color(self, color_name, color_format='bgr'):
        try:
            color = webcolors.name_to_rgb(color_name)
        except ValueError:
            color = IntegerRGB(128, 128, 128)  # Return grey for unknown colors

        if color_format == 'bgr':
            return (color.blue, color.green, color.red)
        elif color_format == 'rgb':
            return (color.red, color.green, color.blue)
        else:
            raise ValueError("Unsupported color format. Use 'rgb' or 'bgr'.")


    def draw_orientation(self, img, size=35, thickness=2):
        max_y, max_x = img.shape[:2]
        x = max(max_x-size-15, 0)
        y = min(15, max_y)
        # Define cube points
        points = np.array([
            [x, y],
            [x + size, y],
            [x + size, y + size],
            [x, y + size],
            [x + int(0.3 * size), y - int(0.3 * size)],
            [x + size + int(0.3 * size), y - int(0.3 * size)],
            [x + size + int(0.3 * size), y + size - int(0.3 * size)],
            [x + int(0.3 * size), y + size - int(0.3 * size)]
        ], dtype=np.int32)

        # Define cube edges
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Front face
            (4, 5), (5, 6), (6, 7), (7, 4),  # Back face
            (0, 4), (1, 5), (2, 6), (3, 7)  # Connecting edges
        ]

        # Draw cube edges
        for edge in edges:
            cv2.line(img, tuple(points[edge[0]]), tuple(points[edge[1]]), (0, 0, 0), thickness)

        # Fill faces with semi-transparent color
        alpha = 0.8
        faces = [
            np.array([points[i] for i in [0, 1, 5, 4]], dtype=np.int32),  # current_upper
            np.array([points[i] for i in [0, 1, 2, 3]], dtype=np.int32),  # current_left
            np.array([points[i] for i in [1, 2, 6, 5]], dtype=np.int32)   # current_front
        ]

        overlay = img.copy()
        cv2.fillPoly(overlay, [faces[0]], color=self._plot_get_color(self.current_upper))
        cv2.fillPoly(overlay, [faces[1]], color=self._plot_get_color(self.current_left))
        cv2.fillPoly(overlay, [faces[2]], color=self._plot_get_color(self.current_front))

        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

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

        self.draw_orientation(image)


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
        self.target_state = CubeState()

        self.last_color_map_s = deque(maxlen=30)
        self.solution = None

        self.MAX_ATEMPS_COLOR_ESTIMATION = 15
        self.n_estimate = 0

    def is_estimation_ok(self, inliners, MIN_INLINERS=12):
        return len(inliners) >= MIN_INLINERS

    def map_cube(self, img, surfaces):
        colors = dict()
        for name, surf in surfaces.items():
            surf_colors = self.clf.estimate_colors(img, surf)
            colors[name] = surf_colors
        return colors

    def update_cube(self, img, proj2d_s):
        alligned_surfaces = align_cube_surfaces(proj2d_s)
        color_map = self.map_cube(img, alligned_surfaces)
        self.update_cube_state(color_map)

    def update_cube_state(self, color_map):
        try:
            if self.n_estimate >= self.MAX_ATEMPS_COLOR_ESTIMATION:
                self.cube_state.reset('all')
                self.target_state.reset('all')
                self.n_estimate = 0

            if self.cube_state.upper == []:
                self.cube_state.update(upper=color_map['upper'], left=color_map['left'], front=color_map['front'])
                self.target_state = self.cube_state.copy_from_state(current_upper='!'+self.cube_state.current_upper,
                                                                    current_front='!'+self.cube_state.current_front,
                                                                    current_left='!'+self.cube_state.current_left)
            elif self.cube_state.down == []:
                try:
                    self.cube_state.update(upper=color_map['upper'], left=color_map['left'], front=color_map['front'])
                    self.n_estimate += 1
                except CubeStateColorError:
                    self.n_estimate += 0.1
                except Exception:
                    self.n_estimate += 1
            else:# orientation update
                self.cube_state.update(upper=color_map['upper'], left=color_map['left'], front=color_map['front'])


            if self.solution is None and self.cube_state.down != []:
                if self.cube_state.check_consistency_full_map() is False:
                    self.cube_state.reset('down')
                    self.n_estimate += 1
                else:
                    try:
                        self.update_solution()
                        self.last_color_map_s.append(color_map)
                    except Exception:
                        print('solution not found resetting cube')
                        self.action_state = []
                        self.cube_state.reset('all')
                        self.n_estimate = 0
            else:
                if self.cube_state._check_orientation_consitency(color_map['upper'][4], color_map['left'][4],color_map['front'][4]):
                    try:
                        for last_color_map in reversed(self.last_color_map_s):
                            rotation = self.cube_state.detect_rotation(last_color_map, color_map)
                            if rotation:
                                self.cube_state = self.cube_state.get_next_cube(rotation)
                                self.update_solution(rotation)
                                self.last_color_map_s.clear()
                    except Exception as e:
                        print(f'GOT ERROR {e}')
                    self.last_color_map_s.append(color_map)
        except Exception as e:
            print(f"An error occurred: {e}")
            traceback.print_exc()
            

    def update_solution(self, rotation=None):
        if rotation is not None and self.solution is not None and rotation == self.solution.split(' ')[0]:
            self.solution = ' '.join(self.solution.split(' ')[1:])
        else:
            solver_notation = self.cube_state.get_kociemba_string_notation()
            print(solver_notation)
            solution = kociemba.solve(solver_notation)
            self.solution = expand_solution(solution)

        if self.cube_state.get_kociemba_string_notation() == STATE_SOLVED:
            self.target_state = self.cube_state.copy_from_state()
            print('cube solved')
        else:
            self.target_state = self.cube_state.get_next_cube(self.solution)


class CubePlanner():
    def __init__(self, K,
                 action_executor = None,
                 WIDTH_CUBE_DETECT = 320):
        self.image = None
        self.WIDTH_CUBE_DETECT = WIDTH_CUBE_DETECT
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f'[CubePlanner] loading cube seg, with {device=}')
        # FastSAM https://huggingface.co/spaces/An-619/FastSAM
        # Used weights
        # https://huggingface.co/spaces/An-619/FastSAM/resolve/main/weights/FastSAM.pt
        self.K = K
        self.dist_coeffs = np.zeros((4, 1))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cube_seg = CubeSegmentation(device=device)
        self.color_cls = ColorClassiferKmeans().load()
        self.cube_solver = CubeSolver(self.color_cls)
        if action_executor is None:
            action_executor = ActionExecutor()
        self.action_executor = action_executor
        self.box, self.rvec, self.tvec, self.midpoints, self.proj_inliner = None, None, None, None, None

        processing_thread = threading.Thread(target=self.estimation_loop, args=())
        processing_thread.start()

    def init_image(self,image, rotate_img):
        image = image.copy()
        if rotate_img:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        self.image = image

    def highlight_cube(self,frame, tvec, rvec):
        cube_points = get_cube_edges()
        imgpts, _ = cv2.projectPoints(cube_points, rvec, tvec, self.K, self.dist_coeffs)
        draw_cube_adaptive_edges(frame, imgpts)

    def plot_solution(self, image, solution):
        color = (0, 255, 0)  # BGR format for green
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.3
        thickness = 1
        text_bottom = image.shape[0] - 10
        cv2.putText(image, '[solution] ' + solution, (7, text_bottom),
                    font, font_scale, color, thickness, cv2.LINE_AA)




    def plot(self, image, plot_bounding_box=False, plot_projection=True, plot_cube_state=True, plot_action=True):
        if self.tvec is not None and self.rvec is not None:
            self.highlight_cube(image, self.tvec, self.rvec)

            if plot_bounding_box:
                draw_bounding_box(image=image, pred=self.box,  color=(0, 0, 255))

            if plot_projection:
                if self.midpoints is not None and self.proj_inliner is not None:
                    highlight_points(points=self.midpoints, projections=self.proj_inliner, image=image)

            if plot_cube_state:
                if self.cube_solver.solution is not None:
                    self.plot_solution(image, self.cube_solver.solution)
                self.cube_solver.cube_state.plot(image)

            if plot_action:
                self.action_executor.plot_action(image)

    def estimation_loop(self):
        while True:
            frame = self.image
            if frame is None:
                continue
            box, _ = self.cube_seg.detect_cube(frame, new_width=self.WIDTH_CUBE_DETECT)
            if box is None:
                self.box = box
                print('could not detect cube')
                continue
            mid_points = self.cube_seg.get_midpoints(frame, box)

            rvec, tvec, midpoints, proj_inliner = estimate_cube_pose(mid_points=mid_points,
                                                                     K=self.K,
                                                                     dist_coeffs=self.dist_coeffs)

            if rvec is not None:
                self.rvec, self.tvec, self.midpoints, self.proj_inliner = rvec, tvec, midpoints, proj_inliner
                try:
                    success, proj2d_s = get_surfaces_Q1_Q2_Q3(rvec=self.rvec,
                                                              tvec=self.tvec,
                                                              K=self.K,
                                                              dist_coeffs=self.dist_coeffs,
                                                              inliners=self.proj_inliner)
                    if success:
                        self.cube_solver.update_cube(img=frame, proj2d_s=proj2d_s)
                except Exception as e:
                    pass
            try:
                self.action_executor.run(self.cube_solver.cube_state, self.cube_solver.target_state, self.cube_solver.solution)
            except Exception as e:
                print(f'[action executor error] {e}')

class ActionExecutor:
    def  __init__(self):
        self.current_action = ""

    def _get_next_action_show_other_side(self, cube_state: CubeState, target_state: CubeState):
        if (cube_state.current_upper == target_state.current_upper[1:]):
            return 'flip'
        if (cube_state.current_upper == target_state.current_left[1:]):
            return 'flip'
        if (cube_state.current_front == target_state.current_front[1:]):
            return 'rotate_left'
        return ""

    def get_next_action(self, cube_state: CubeState, target: CubeState, solution=""):
        if cube_state == target:
            return None
        if cube_state.current_upper == "" or target.current_upper == "":
            print('warning got empty state')
            return None
        if not solution:
            return self._get_next_action_show_other_side(cube_state, target)
        else:
            side = solution[0]
            if cube_state.current_upper == cube_state.side2midpoint[side]:
                if "'" in solution[:2]:
                    return 'rotate_upper_left'
                else:
                    return 'rotate_upper_right'
            else:
                side_upper = cube_state.midpoint2side[cube_state.current_upper]
                side_left = cube_state.midpoint2side[cube_state.current_left]
                side_front = cube_state.midpoint2side[cube_state.current_front]
                side2opposite = {"U": "D", "D":"U", "L":"R", "R":"L", "F":"B", "B":"F"}
                if side2opposite[side_upper] == side:
                    return 'flip'
                elif side_left == side:
                    return'flip'
                elif side_front == side:
                    return 'rotate_right'
                else:
                    return 'rotate_left'


    def run(self, cube_state: CubeState, target_state: CubeState, solution:str):
        self.current_action = self.get_next_action(cube_state, target_state, solution)

    def plot_action(self, image):
        height, width = image.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (255, 0, 0)
        font_thickness = 2
        text_size = cv2.getTextSize(self.current_action, font, font_scale, font_thickness)[0]
        text_x = (width - text_size[0]) // 2
        text_y = 30
        cv2.putText(image, self.current_action, (text_x, text_y), font, font_scale, font_color, font_thickness)