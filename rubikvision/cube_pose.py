import cv2
import numpy as np
from scipy.spatial import distance_matrix


def shift_points(points, shift):
    if isinstance(points, np.ndarray):
        points += np.array(shift)
    else:
        points = [(p[0] + shift[0], p[1] + shift[1]) for p in points]
    return points

def calculate_midpoint(contour):
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0
    return (cX, cY)

def get_cube_edges():
    cube_length = 0.057
    cube_points = np.float32([
        [0, 0, 0], [0, cube_length, 0], [cube_length, cube_length, 0], [cube_length, 0, 0],
        [0, 0, cube_length], [0, cube_length, cube_length],
        [cube_length, cube_length, cube_length], [cube_length, 0, cube_length]
    ])
    return cube_points

def get_cube_surfaces(as_np_arrays=False):
    #     +-------+
    #    /   Q3   /
    #   /       / |
    #  +-------+  |
    #  |       |Q2|
    #  |   Q1  |  |
    #  |       | /
    #  +-------+/

    C = 0.057/3
    Q13d = (
        (C/2,0,C/2),
        (C/2+C,0,C/2),
        (C/2+2*C,0,C/2),

        (C/2,0,  C+C/2),
        (C/2+C,0,  C+C/2),
        (C/2+2*C,0,C+C/2),

        (C/2,0,  2*C+C/2),
        (C/2+C,0,  2*C+C/2),
        (C/2+2*C,0,2*C+C/2),
    )
    Q23d = (
        (C/2,C/2,3*C),
        (C/2+C,C/2,3*C),
        (C/2+2*C,C/2,3*C),

        (C/2,C/2 + C,3*C),
        (C/2+C,C/2 + C,3*C),
        (C/2+2*C,C/2 + C,3*C),

        (C/2, C / 2 + 2*C, 3 * C),
        (C/2+C, C / 2 + 2*C, 3 * C),
        (C/2+2*C, C / 2 + 2*C, 3 * C),
    )

    Q33d = (
        (C*3, C/2,C/2),
        (C*3, C/2+C,C/2),
        (C*3, C/2+C*2,C/2),

        (C*3,C/2,  C+C/2),
        (C*3,C/2+C,  C+C/2),
        (C*3,C/2+C*2,C+C/2),

        (C*3,C/2,  2*C+C/2),
        (C*3,C/2+C,  2*C+C/2),
        (C*3,C/2+C*2,2*C+C/2),
    )
    if as_np_arrays:
        Q13d = np.array(Q13d, dtype=np.float32)
        Q23d = np.array(Q23d, dtype=np.float32)
        Q33d = np.array(Q33d, dtype=np.float32)
    return Q13d, Q23d, Q33d


def rate_proj_points(points2d, points2d_proj, deviation=2.):
    points2d = np.array(points2d, dtype=np.float32)
    points2d_proj = np.array(points2d_proj, dtype=np.float32)
    dist_matrix = distance_matrix(points2d, points2d_proj)
    mask = dist_matrix <= deviation

    best_matches = np.argmin(np.where(mask, dist_matrix, np.inf), axis=1)
    inliers = np.zeros_like(mask, dtype=bool)
    for i, best_match in enumerate(best_matches):
        if mask[i, best_match]:
            inliers[i, best_match] = True

    count = np.sum(np.any(inliers, axis=0))
    inlier_points = None
    if count > 0:
        inlier_distances = dist_matrix[inliers]
        mean_error = np.mean(inlier_distances)
        inlier_indices = np.where(np.any(inliers, axis=0))[0]
        inlier_points = points2d_proj[inlier_indices]
        assert len(inlier_points) == count

        mask = np.ones(len(points2d_proj), dtype=bool)
        mask[inlier_indices] = False
        outlier_points = points2d_proj[mask]
        assert len(outlier_points) + len(inlier_points) == len(points2d_proj)

        max = np.ceil(np.max(inlier_points)).astype(int)
        image = np.zeros((max+1, max+1), dtype=np.uint8)
        for point in inlier_points:
            image[int(point[1]), int(point[0])] = 255  # Set the point to white
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = np.vstack(contours)
        points_inside_projection = 0
        for p in outlier_points.astype(np.int32):
            result = cv2.pointPolygonTest(contour, (int(p[0]), int(p[1])), False)
            if result>=0:
                points_inside_projection += 1
    else:
        mean_error = np.inf
        points_inside_projection = np.inf

    # Assuming that the last part of the points is the upper part of the cube, only necessary for orrientation
    N = len(points2d_proj)//3
    mid_1 = np.mean(points2d_proj[0:N], axis=0)
    mid_2 = np.mean(points2d_proj[N:N*2], axis=0)
    mid_3 = np.mean(points2d_proj[N*2:], axis=0)
    top_oriented = mid_3[0] < mid_2[0] and mid_3[0]<mid_1[0]

    return count, mean_error, points_inside_projection, top_oriented, inlier_points

def rate_proj_result(points3dall, points2d, rvec, tvec, K, dist_coeffs, deviation=2, verbose=False):
    points2d = np.array(points2d, np.float32)
    points2d_proj = reproject_points(points3d=points3dall, rvec=rvec, tvec=tvec, K=K,dist_coeffs=dist_coeffs)

    res =  rate_proj_points(points2d=points2d, points2d_proj=points2d_proj, deviation=deviation)
    if verbose:
        print('points2d', points2d)
        print('points2d_proj', points2d_proj)
        print(res)
    return res

def reproject_points(points3d, rvec, tvec, K, dist_coeffs):
    points2d, _ = cv2.projectPoints(points3d, rvec, tvec, K, dist_coeffs)
    return points2d.squeeze()

def find_surface_lowest_left_point(points):
    hull = cv2.convexHull(np.array(points))
    epsilon = 0.02 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True).squeeze()
    lowest_2_points = approx[np.argsort(-approx[:, 1])][:2]
    lowest_left_point = lowest_2_points[np.argsort(lowest_2_points[:, 0])][0]
    return tuple(lowest_left_point)

def get_ordered_rubik_points(points):
    lowest_left_point = find_surface_lowest_left_point(points)
    points = np.array(points)
    center = points[0]
    border = points[1:]
    vectors = border - center
    degs = (np.arctan2(vectors[:, 1], vectors[:, 0]) * 180 / np.pi) % 360
    sorted_points = border[np.argsort(-degs)]
    idx = sorted_points.tolist().index(list(lowest_left_point))
    points_sorted = [
        tuple(sorted_points[ (idx+0) % 8]),
        tuple(sorted_points[ (idx+1) % 8]),
        tuple(sorted_points[ (idx+2) % 8]),

        tuple(sorted_points[ (idx+7) % 8]),
        tuple(center),
        tuple(sorted_points[ (idx+3) % 8] ),

        tuple(sorted_points[(idx + 6) % 8] ),
        tuple(sorted_points[(idx + 5) % 8] ),
        tuple(sorted_points[(idx + 4) % 8] ),
        ]
    return points_sorted

def list_to_grid(lst):
    """Converts a flat list to a 3x3 grid."""
    return [lst[i:i+3] for i in range(0, len(lst), 3)]

def grid_to_list(grid):
    """Converts a 3x3 grid to a flat list."""
    return [item for row in grid for item in row]

def rotate_90_clockwise(matrix):
    return [list(reversed(col)) for col in zip(*matrix)]

def reverse_rows(matrix):
    return [row[::-1] for row in matrix]

def get_all_rotations_and_orders_of_surface_points(points):
    grids = []
    grid = list_to_grid(points)
    for _ in range(4):
        grid = rotate_90_clockwise(grid)
        grids.append(grid)
    for n in range(4):
        grids.append(reverse_rows(grids[n]))
    return [grid_to_list(g) for g in grids]


def remove_nearby_points(points, threshold=5):
    result = []
    for point in points:
        if not any(np.linalg.norm(np.array(p) - np.array(point))< threshold for p in result):
            result.append(point)
    return result

def find_rubik_surface(points, search=3, CENTER_DIST=10):
    points = remove_nearby_points(points, threshold=3)
    points = np.array(points)
    midpoints = []
    dist_matrix = distance_matrix(points, points)
    for i, center in enumerate(points):
        nearest = np.argsort(dist_matrix[i])
        if len(nearest) < 8:
            continue

        threshold = dist_matrix[i][nearest[3]] / 2
        nearest_threshold = nearest[ dist_matrix[i][nearest] > threshold]

        if len(nearest_threshold) < 8:
            continue

        neighbors_idxs = [np.hstack([nearest_threshold[:7],  nearest_threshold[7+i]])
                          for i in range(min(search, len(nearest_threshold)-7))]
        neighbors_idxs += [np.hstack([nearest_threshold[:6],  nearest_threshold[6+i+1:6+i+2+1]])
                          for i in range(min(search, len(nearest_threshold)-7))]

        neighbors_candidates = [points[idxs] for idxs in neighbors_idxs if len(idxs)==8]
        for neighbors in neighbors_candidates:
            center_dist = np.linalg.norm(np.mean(neighbors, axis=0) - center)
            if center_dist < CENTER_DIST:
                print(center, center_dist)
                hull = cv2.convexHull(neighbors)
                epsilon = 0.03 * cv2.arcLength(hull, True)
                approx = cv2.approxPolyDP(hull, epsilon, True)
                if len(approx) <= 4:
                    midpoints.append(np.vstack([center, neighbors]))
    return midpoints

def _filter_contour_outliners(areas):
    areas = np.array(areas)
    med = np.median(areas)
    deviation = np.abs(areas / med)
    MIN_AREA_DEV = 0.3
    MAX_AREA_DEV = 2.5
    outliners = np.logical_and(deviation>MIN_AREA_DEV, deviation< MAX_AREA_DEV)
    return outliners

def get_square_contours(annotations):
    annotations = annotations.cpu().numpy()
    ret = []
    areas = []
    for mask in annotations:
        if isinstance(mask, dict):
            mask = mask['segmentation']
        annotation = mask.astype(np.uint8)
        contours, _ = cv2.findContours(annotation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = np.vstack(contours)
        ret.append(cnt)
        areas.append(cv2.contourArea(cnt))
    filter = _filter_contour_outliners(areas)
    ret = [cnt for n, cnt in enumerate(ret) if filter[n]]
    return ret

def estimate_cube_pose(mid_points, K, dist_coeffs):
    if mid_points == [] or len(mid_points) < 4:
        print('warning got empty midpoints')
        return None, None, None, None
    clusters = find_rubik_surface(mid_points)
    if len(clusters) == 0:
        print('warning got empty midpoints')
        return None, None, None, None
    ordered_clusters = [get_ordered_rubik_points(clust) for clust in clusters]
    rotated_clusters = [clust for clust_er in ordered_clusters
                        for clust in get_all_rotations_and_orders_of_surface_points(clust_er)]
    Q13d, Q23d, Q33d = get_cube_surfaces(as_np_arrays=True)
    estimates = []
    for clust in rotated_clusters:
        for points3d in [Q13d, Q23d, Q33d]:
            success, rvec, tvec = cv2.solvePnP(points3d, np.array(clust, dtype=np.float32),
                                                                        K, dist_coeffs, flags=cv2.SOLVEPNP_IPPE)
            if success:
                estimates.append((rvec, tvec))

    if estimates == []:
        print('warning could not estimate cube')
        return None, None, None, None

    points3dall = np.vstack([Q13d, Q23d, Q33d], dtype=np.float32)
    ratings = np.array([rate_proj_result(points3dall=points3dall,
                                         points2d=mid_points,
                                         rvec=e[0],
                                         tvec=e[1],
                                         K=K,
                                         dist_coeffs=dist_coeffs)[:3]
                        for e in estimates
                        ])
    sorting_key = np.lexsort((ratings[:, 1],ratings[:, 2], -ratings[:, 0], ))
    sorted_estimates = [estimates[i] for i in sorting_key]

    best_estimate = sorted_estimates[0]
    best_res = rate_proj_result(points3dall=points3dall,
                     points2d=mid_points,
                     rvec=best_estimate[0],
                     tvec=best_estimate[1],
                     K=K,
                     dist_coeffs=dist_coeffs, verbose=False)
    best_projection = best_res[-1]
    return best_estimate[0], best_estimate[1], mid_points, best_projection

def get_surfaces_Q1_Q2_Q3(rvec, tvec, K, dist_coeffs, inliners, MIN_MATCH_FOR_SUCC=3):
    proj2d_s = []
    success = True
    for n, Q3d in enumerate(get_cube_surfaces(as_np_arrays=True)):
        proj = reproject_points(Q3d, rvec, tvec, K, dist_coeffs)
        diff = np.abs(inliners[:, np.newaxis, :] - proj[np.newaxis, :, :])
        matches = np.sum(np.all(diff <= 0.01, axis=2))
        success = (matches >= MIN_MATCH_FOR_SUCC) and success
        proj2d_s.append(proj)
    return success, proj2d_s
