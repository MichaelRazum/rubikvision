import pytest
import os
from matplotlib import pyplot as plt, cm
import cv2
from rubikvision.cube_pose import _filter_contour_outliners, find_rubik_surface, \
    find_surface_lowest_left_point, get_ordered_rubik_points, list_to_grid, grid_to_list, rotate_90_clockwise, \
    reverse_rows, rate_proj_points, estimate_cube_pose, get_cube_edges, get_surfaces_Q1_Q2_Q3
import numpy as np

from rubikvision.cube_detection import draw_cube_adaptive_edges, highlight_points, extract_cube
from rubikvision.cube_solver import _flip_list


def __plot_midpoints(mid_points, cluster_list=None):
    if cluster_list is None:
        cluster_list = np.zeros(len(mid_points))
    # Unzip the list of tuples into two lists: x and y coordinates
    x_coords, y_coords = zip(*mid_points)

    # Create a color map for the clusters
    unique_clusters = list(set(cluster_list))
    num_clusters = len(unique_clusters)
    colors = cm.rainbow(np.linspace(0, 1, num_clusters))
    cluster_color_map = {cluster: colors[i] for i, cluster in enumerate(unique_clusters)}

    # Create a scatter plot
    fig, ax = plt.subplots()
    sc = ax.scatter(x_coords, y_coords, c=[cluster_color_map[cluster] for cluster in cluster_list], marker='o')

    # Invert the y-axis to match OpenCV's coordinate system
    plt.gca().invert_yaxis()

    # Add titles and labels
    plt.title('Mid Points Plot')
    plt.xlabel('X Coordinates')
    plt.ylabel('Y Coordinates')

    # Annotate points with coordinates
    annot = ax.annotate("", xy=(0, 0), xytext=(10, 10),
                        textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    def update_annot(ind):
        pos = sc.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        text = f"({int(pos[0])}, {int(pos[1])})"
        annot.set_text(text)
        annot.get_bbox_patch().set_alpha(0.4)

    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = sc.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)

    plt.grid(True)
    plt.show()

def test_flip_list():
    list = [0,1,2,
            3,4,5,
            6,7,8]
    assert _flip_list(list, 8) == _flip_list(list, 0) == [0, 3, 6,
                                                          1, 4, 7,
                                                          2, 5, 8]

    assert _flip_list(list, 6) == _flip_list(list, 2) == [8, 5, 2,
                                                          7, 4, 1,
                                                          6, 3, 0]

    assert _flip_list(list, 7) == _flip_list(list, 1) == [2, 1, 0,
                                                          5, 4, 3,
                                                          8, 7, 6, ]

    assert _flip_list(list, 5) == _flip_list(list, 3) == [6, 7, 8,
                                                          3, 4, 5,
                                                          0, 1, 2, ]


def test_filter_contour_outliners():
    areas = [749.5, 948.5, 722.5, 829.0, 811.5, 718.0, 784.5, 551.0, 607.5, 668.5, 762.0, 670.0, 714.5, 625.5, 578.5, 511.0,
             433.5, 690.5, 404.5, 646.5, 380.5, 417.0, 367.5, 416.5, 327.5, 291.5, 785.5, 23689.0]
    got = _filter_contour_outliners(areas=areas)
    want = np.array([True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True,
         True, True, True, True, True, True, True, True, True, False])
    assert (got == want).all()

def test_rate_points():
    points = [(0,0), (0,1), (1,0)]
    points_proj = [(0,1), (5,5)]
    rating = rate_proj_points(points, points_proj, deviation=0.1)
    assert rating[0] == 1
    assert rating[1] == 0
    assert rating[2] == 0

    points = [(0,0), (0,1), (1,0), (5,0)]
    points_proj = [(0,0), (0,1), (1,0), (0.5,0.5)]
    rating = rate_proj_points(points, points_proj, deviation=0.1)
    assert rating[0] == 3
    assert rating[1] == 0
    assert rating[2] == 1

    points = [(0,0), (0,1), (1,0), (5,0)]
    points_proj = [(0,0), (0,1), (1,0), (0.5,0.5)]
    rating = rate_proj_points(points, points_proj, deviation=0.1)
    assert rating[0] == 3
    assert rating[1] == 0
    assert rating[2] == 1

    points2d = [
        [146, 210],
        [162, 234],
        [175, 266],
        [219, 247],
        [212, 158],
        [173, 172],
        [247, 199],
        [229, 177],
        [265, 262],
        [133, 187],
        [271, 292],
        [225, 280],
        [195, 325],
        [259, 229],
        [188, 194],
        [206, 216],
        [233, 308],
        [185, 296]
    ]

    points2d_proj = [
        [211.86, 157.65],
        [173.12, 172.23],
        [132.62, 187.48],
        [228.75, 177.43],
        [188.57, 193.17],
        [146.48, 209.65],
        [247.29, 199.14],
        [205.56, 216.18],
        [161.74, 234.08],
        [260.5, 228.3],
        [219.07, 246.07],
        [175.56, 264.72],
        [266.57, 260.93],
        [227.19, 278.64],
        [185.93, 297.19],
        [272.11, 290.75],
        [234.6, 308.33],
        [195.37, 326.72],
        [117.99, 212.19],
        [129.79, 243.68],
        [140.63, 272.61],
        [130.98, 235.28],
        [142.74, 267.19],
        [153.51, 296.38],
        [145.27, 260.71],
        [156.94, 292.96],
        [167.56, 322.34]
    ]
    rating = rate_proj_points(points2d, points2d_proj, deviation=2)
    assert rating[0] == 17
    assert rating[2] == 1

    points2d = [
        [109, 254],
        [137, 226],
        [148, 257],
        [179, 228],
        [188, 260],
        [100, 192],
        [141, 194],
        [132, 144],
        [122, 283],
        [155, 168],
        [182, 195],
        [158, 285],
        [116, 167],
        [196, 289],
        [207, 146],
        [208, 214],
        [220, 188],
        [195, 169],
        [223, 274],
        [169, 145],
        [231, 163],
        [227, 220],
        [238, 195],
        [216, 246],
        [98, 224]
    ]

    points2d_proj = [
        [131.69, 144.09],
        [116.54, 166.95],
        [99.866, 192.12],
        [168.91, 144.94],
        [155.39, 168.1],
        [140.51, 193.6],
        [206.91, 145.81],
        [195.11, 169.26],
        [182.09, 195.12],
        [229.56, 162.82],
        [219.04, 186.71],
        [207.46, 213.02],
        [235.87, 193.99],
        [226.08, 218.33],
        [215.34, 245.02],
        [241.7, 222.78],
        [232.56, 247.44],
        [222.56, 274.37],
        [97.622, 222.52],
        [110.22, 254.02],
        [121.78, 282.91],
        [138.35, 224.35],
        [149.31, 256.12],
        [159.35, 285.23],
        [180.02, 226.24],
        [189.26, 258.27],
        [197.72, 287.6]
    ]

    rating = rate_proj_points(points2d, points2d_proj, deviation=2)
    assert rating[0] == 20
    assert rating[2] == 1


    points2d = [
        [175, 266],
        [147, 210],
        [219, 247],
        [162, 234],
        [173, 172],
        [225, 280],
        [265, 262],
        [213, 157],
        [271, 292],
        [229, 177],
        [259, 229],
        [133, 187],
        [195, 325],
        [247, 199],
        [185, 296],
        [189, 194],
        [206, 216],
        [233, 308]
    ]

    points2d_proj = [
        [133.16, 187.37],
        [146.83, 209.71],
        [161.9, 234.31],
        [173.61, 171.96],
        [188.89, 193.05],
        [205.7, 216.25],
        [212.36, 157.19],
        [229.1, 177.14],
        [247.46, 199.02],
        [234.67, 166.68],
        [251.87, 186.4],
        [270.68, 207.97],
        [241.35, 197.94],
        [258.1, 218.18],
        [276.36, 240.23],
        [247.5, 226.78],
        [263.83, 247.39],
        [281.57, 269.77],
        [175.77, 264.92],
        [186.44, 297.11],
        [196.14, 326.38],
        [219.26, 246.12],
        [227.67, 278.46],
        [235.33, 307.93],
        [260.73, 228.19],
        [267.07, 260.63],
        [272.85, 290.25]
    ]

    rating = rate_proj_points(points2d, points2d_proj, deviation=2)
    cluster_list = np.ones(len(points2d+points2d_proj))
    cluster_list[:len(points2d)] = 0
    __plot_midpoints(points2d+points2d_proj, cluster_list=cluster_list)
    assert rating[0] == 14
    assert rating[2] == 1

@pytest.mark.parametrize('points, lowest_left', [
    ([(33, 67),(49, 42),(65, 19),(74, 69),(88, 43),(102, 20),(115, 70),(128, 44),(140, 21)],     (33,67)),
    ([(31, 99), (42, 129), (55, 158), (70, 101),(81, 132), (91, 160),(112, 103),(121, 135),(129, 164)],     (55,158)),
    ([(26, 88),(38, 119),(48, 146),(57, 99),(67, 130),(76, 157),(89, 111),(97, 141),(107, 170)],     (48,146)),
    ([(36, 60), (63, 40), (67, 69), (87, 21), (92, 48), (99, 80), (117, 29), (124, 58), (148, 37)],     (36,60)),
] )
def test_get_lowest_left_point(points, lowest_left):
    lowest_left_point = find_surface_lowest_left_point(points)
    # __plot_midpoints(points)
    assert lowest_left_point == lowest_left


@pytest.mark.parametrize('points, ordered_points', [
    ([(88, 43), (33, 67),(49, 42),(65, 19),(74, 69),(102, 20),(115, 70),(128, 44),(140, 21)],
     [(33,67), (74,69), (115,70), (49,42), (88,43), (128,44), (65,19), (102,20), (140,21)]),
    ([(81, 132), (31, 99), (42, 129), (55, 158), (70, 101),(91, 160),(112, 103),(121, 135),(129, 164)],
     [(55,158), (91,160), (129,164), (42,129), (81,132), (121,135), (31, 99), (70,101), (112,103)]),
    ([(67, 130), (26, 88),(38, 119),(48, 146),(57, 99),(76, 157),(89, 111),(97, 141),(107, 170)],
    [(48, 146), (76, 157), (107, 170), (38, 119), (67, 130), (97, 141), (26, 88), (57, 99), (89, 111)]),
] )
def test_get_ordered_rubik_points(points, ordered_points):
    ordered_points_got = get_ordered_rubik_points(points)
    # __plot_midpoints(points)
    assert ordered_points_got == ordered_points


def test_find_rubik_surface(plot=False):
    def assert_and_plot(mid_points, clusters, expected_clusters, plot, MAX_CLUSTERS):
        assert len(clusters) <= MAX_CLUSTERS, 'To many candidates found'
        clusters = [set(tuple(c) for c in clust) for clust in clusters]
        if plot:
            for clust in clusters:
                __plot_midpoints(mid_points, [1 if p in clust else 0 for p in mid_points])
        for clust in  expected_clusters:
            assert set(tuple(c) for c in clust) in clusters, f'cluster not found {clust}'

    mid_points = [(42, 129), (70, 101), (81, 132), (112, 103), (121, 135), (33, 67), (74, 69), (65, 19), (55, 158),
                  (88, 43), (115, 70), (91, 160), (49, 42), (129, 164), (140, 21), (141, 89), (153, 63), (128, 44),
                  (156, 149), (102, 20), (164, 38), (160, 95), (171, 70), (149, 121), (31, 99)]
    clusters = find_rubik_surface(mid_points)
    expected_clusters = [{(33, 67), (49, 42), (65, 19), (74, 69), (88, 43), (102, 20), (115, 70), (128, 44), (140, 21)},
                         {(31, 99), (42, 129), (55, 158), (70, 101), (81, 132), (91, 160), (112, 103), (121, 135),
                          (129, 164)}]
    assert_and_plot(mid_points=mid_points,clusters=clusters,expected_clusters=expected_clusters,plot=plot,MAX_CLUSTERS=10)
    #Tets 7
    mid_points = [(150, 141), (104, 102), (24, 80), (134, 83), (142, 113), (169, 93), (70, 152), (36, 110), (114, 133),
                  (65, 35), (85, 77), (94, 166), (48, 93), (93, 18), (144, 40), (174, 121), (59, 123), (34, 53),
                  (163, 63), (118, 29), (73, 107), (84, 137), (59, 65), (123, 162), (115, 59), (90, 47)]
    clusters = find_rubik_surface(mid_points)
    expected_clusters = [{(104, 102), (114, 133), (123, 162), (134, 83), (142, 113), (150, 141), (163, 63), (169, 93), (174, 121)},
                         { (34, 53),(59, 65),(65, 35),(85, 77),(90, 47),(93, 18),(115, 59),(118, 29),(144, 40)}]
    assert_and_plot(mid_points=mid_points,clusters=clusters,expected_clusters=expected_clusters,plot=plot,MAX_CLUSTERS=10)
    #Tets 10
    mid_points = [(26, 88), (97, 141), (36, 60), (157, 139), (120, 103), (76, 157), (89, 111), (38, 119), (57, 99),
                  (128, 135), (135, 163), (172, 89), (167, 59), (176, 117), (151, 111), (124, 58), (67, 69), (87, 21),
                  (99, 80), (144, 81), (148, 37), (92, 48), (63, 40), (117, 29), (67, 130), (48, 146), (107, 170)]
    clusters = find_rubik_surface(mid_points)
    expected_clusters = [{(120, 103),(128, 135),(135, 163),(144, 81),(151, 111),(157, 139),(167, 59),(172, 89),(176, 117)},
                         {(36, 60), (63, 40), (67, 69), (87, 21), (92, 48), (99, 80), (117, 29), (124, 58), (148, 37)},
                         {(26, 88),(38, 119),(48, 146),(57, 99),(67, 130),(76, 157),(89, 111),(97, 141),(107, 170)}]
    assert_and_plot(mid_points=mid_points,clusters=clusters,expected_clusters=expected_clusters,plot=plot,MAX_CLUSTERS=10)

    # Added nearby point
    mid_points = [(26, 88), (97, 141), (36, 60), (157, 139), (120, 103), (76, 157), (89, 111), (38, 119), (57, 99),
                  (128, 135), (135, 163), (172, 89), (167, 59), (176, 117), (151, 111), (124, 58), (67, 69), (87, 21),
                  (99, 80), (144, 81), (148, 37), (92, 48), (63, 40), (117, 29), (67, 130), (48, 146), (107, 170),
                  (152,112)]
    clusters = find_rubik_surface(mid_points)
    expected_clusters = [{(120, 103),(128, 135),(135, 163),(144, 81),(151, 111),(157, 139),(167, 59),(172, 89),(176, 117)},
                         {(36, 60), (63, 40), (67, 69), (87, 21), (92, 48), (99, 80), (117, 29), (124, 58), (148, 37)},
                         {(26, 88),(38, 119),(48, 146),(57, 99),(67, 130),(76, 157),(89, 111),(97, 141),(107, 170)}]
    assert_and_plot(mid_points=mid_points,clusters=clusters,expected_clusters=expected_clusters,plot=plot,MAX_CLUSTERS=13)

    mid_points = [(356, 177), (387, 252), (355, 239), (389, 222), (356, 127), (324, 166), (404, 121), (325, 198), (325, 226),
     (413, 248), (356, 210), (436, 168), (433, 200), (430, 229), (391, 190), (446, 212), (418, 187), (419, 146),
     (416, 219), (374, 112), (387, 136), (453, 152), (436, 130), (333, 143), (366, 153), (449, 183), (328, 155),
     (400, 163), (399, 221), (306, 222)]
    expected_clusters = [{(324, 166),(325, 198),(325, 226),(355, 239),(356, 177),(356, 210),(387, 252),(389, 222),(391, 190)}]
    clusters = find_rubik_surface(mid_points)
    assert_and_plot(mid_points=mid_points,clusters=clusters,expected_clusters=expected_clusters,plot=plot,MAX_CLUSTERS=13)
    mid_points = [(414, 170), (380, 159), (410, 231), (442, 243), (380, 191), (379, 220), (411, 202), (410, 120),
                  (426, 105), (491, 122), (476, 138), (457, 113), (492, 159), (483, 219), (507, 143), (502, 174),
                  (448, 181), (497, 202), (468, 239), (488, 191), (442, 129), (445, 213), (477, 178), (472, 210),
                  (459, 154), (390, 136), (367, 95), (423, 145), (454, 212), (424, 144), (458, 180)]
    expected_clusters = [{(423, 145), (476, 138), (448, 181), (410, 120), (457, 113), (459, 154), (426, 105), (491, 122), (442, 129)}]
    clusters = find_rubik_surface(mid_points)
    assert_and_plot(mid_points=mid_points, clusters=clusters, expected_clusters=expected_clusters, plot=plot,
                    MAX_CLUSTERS=13)


def test_list_to_grid():
    assert list_to_grid([6, 7, 8, 3, 4, 5, 0, 1, 2]) == [
        [6, 7, 8],
        [3, 4, 5],
        [0, 1, 2]
    ]

def test_grid_to_list():
    assert grid_to_list([
        [6, 7, 8],
        [3, 4, 5],
        [0, 1, 2]
    ]) == [6, 7, 8, 3, 4, 5, 0, 1, 2]

def test_rotate_90_clockwise():
    assert rotate_90_clockwise([
        [6, 7, 8],
        [3, 4, 5],
        [0, 1, 2]
    ]) == [
        [0, 3, 6],
        [1, 4, 7],
        [2, 5, 8]
    ]

    assert rotate_90_clockwise(rotate_90_clockwise(rotate_90_clockwise(rotate_90_clockwise([
        [0, 3, 6],
        [1, 4, 7],
        [2, 5, 8]
    ])))) == [
        [0, 3, 6],
        [1, 4, 7],
        [2, 5, 8]
    ]

def test_reverse_rows():
    assert reverse_rows([
        [6, 7, 8],
        [3, 4, 5],
        [0, 1, 2]
    ]) == [
        [8, 7, 6],
        [5, 4, 3],
        [2, 1, 0]
    ]

@pytest.mark.skip(reason='test only for debugging')
@pytest.mark.parametrize('img_id', [26,37,38])
def test_cube_mid_points(img_id, cube_seg, datadir,  plot=True):
    img_path = os.path.join(datadir, f'cubepic_{img_id}.jpg')
    img = cv2.imread(img_path)

    cv2.imshow('Rubik\'s Cube Projection (5.7 cm)', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    box, _ = cube_seg.detect_cube(img)
    roi, x_min, y_min = extract_cube(img, box)
    cv2.imwrite(os.path.join(datadir, f'bad_{img_id}.jpg'), roi)
    mid_points = cube_seg.get_midpoints(img, box)
    assert len(mid_points) > 9
    highlight_points(points=mid_points, projections=[], image=img)
    cv2.imshow('Rubik\'s Cube Projection (5.7 cm)', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


@pytest.mark.parametrize('img_id', range(14))
def test_cube_pose_estimation_happy_path(img_id, cube_seg, datadir,  plot=False):
    # https://www.calibdb.net/
    # Logitech C922 PRO
    K = np.array([[632.11326486, 0., 316.16980761],
                  [0., 630.54696352, 233.72252151],
                  [0., 0., 1.]])
    dist_coeffs = np.zeros((4, 1))
    img_path = os.path.join(datadir, f'cubepic_{img_id}.jpg')
    img = cv2.imread(img_path)
    box, _ = cube_seg.detect_cube(img)
    mid_points = cube_seg.get_midpoints(img, box)
    rvec, tvec,_, _= estimate_cube_pose(mid_points, K, dist_coeffs)
    if plot:
        cube_points = get_cube_edges()
        imgpts, _ = cv2.projectPoints(cube_points, rvec, tvec, K, dist_coeffs)
        img_with_cube = draw_cube_adaptive_edges(img.copy(), imgpts)
        cv2.imshow('Rubik\'s Cube Projection (5.7 cm)', img_with_cube)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def test_get_surface2proj():
    K = np.array([[632.11326486, 0., 316.16980761],
                  [0., 630.54696352, 233.72252151],
                  [0., 0., 1.]])
    dist_coeffs = np.zeros((4, 1))
    tvec = np.array([[-0.096714],
                     [-0.049903],
                     [0.31287]])
    rvec = np.array([[0.59608],
                     [1.6666],
                     [0.56853]])
    inliners = np.array(
        [[131.74, 144.25], [116.59, 167.28], [99.912, 192.63], [169.12, 145.1], [155.62, 168.42], [140.74, 194.12],
         [207.28, 145.97], [195.5, 169.59], [182.51, 195.65], [230.02, 163.01], [219.53, 187.07], [207.98, 213.57],
         [236.33, 194.23], [226.58, 218.73], [215.87, 245.61], [233.06, 247.87], [223.1, 274.98], [97.699, 223.16],
         [110.36, 254.7], [121.96, 283.61], [138.61, 225], [149.61, 256.8], [159.68, 285.93], [180.46, 226.89]], )


    success ,surf2proj = get_surfaces_Q1_Q2_Q3(rvec=rvec, tvec=tvec, K=K, dist_coeffs=dist_coeffs, inliners=inliners)
    assert success == True
