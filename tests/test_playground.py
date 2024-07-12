@pytest.mark.parametrize('img_id', [9])
def test_cube_pose_estimation_proof_of_concept(img_id, cube_seg):
    K = np.array([[632.11326486, 0., 316.16980761],
                  [0., 630.54696352, 233.72252151],
                  [0., 0., 1.]])
    dist_coeffs = np.zeros((4, 1))

    # Estimate point surfaces
    img_path = os.path.join(datadir, f'cubepic_{img_id}.jpg')
    img = cv2.imread(img_path)



    Q1 = [(167, 244), (205, 243), (241, 242), (158, 216), (196, 215), (234, 213), (147, 185), (186, 184), (225, 183)]
    Q2 = [(147, 156),  (186, 155), (226, 153), (158, 134),(195, 132), (232, 131), (166, 114),(202, 113), (237, 111)  ]
    Q13d, Q23d, Q33d = get_cube_surfaces()
    success, rotation_vector, translation_vector = cv2.solvePnP(
        np.array(np.vstack([Q13d]), dtype=np.float32), np.array(np.vstack([Q1]), dtype=np.float32), K, dist_coeffs, flags=cv2.SOLVEPNP_IPPE)

    def reproject_points(points3d, rvec, tvec, camera_matrix):
        points2d, _ = cv2.projectPoints(points3d, rvec, tvec, camera_matrix, dist_coeffs)
        return points2d.squeeze()
    points3dall = np.vstack([Q13d, Q23d, Q33d]).astype(np.float32)
    # Reproject the original points
    reprojected_points = reproject_points(points3dall, rotation_vector, translation_vector, K, )

    # PLOT
    if True:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img_rgb)
        for point in Q1+Q2:
            x, y = point
            ax.plot(x, y, 'ro', markersize=10)  # 'ro' means red circle
        # Plot reprojected points
        for point in reprojected_points:
            x, y = point
            ax.plot(x, y, 'bo', markersize=8, label='Reprojected points')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('Iterative ApproxPolyDP Hexagon')
        plt.show()



def test_permute_cluster():
    clust = [(167, 244), (205, 243), (241, 242), (158, 216), (196, 215), (234, 213), (147, 185), (186, 184), (225, 183)]
    get_rotated_points(clust)

@pytest.mark.parametrize('img_id', [1])
def test_cube_pose_estimation_proof_of_concept2(img_id, cube_seg):
    K = np.array([[632.11326486, 0., 316.16980761],
                  [0., 630.54696352, 233.72252151],
                  [0., 0., 1.]])
    dist_coeffs = np.zeros((4, 1))

    # Estimate point surfaces
    img_path = os.path.join(datadir, f'cubepic_{img_id}.jpg')
    img = cv2.imread(img_path)

    img_path = os.path.join(datadir, f'cubepic_{img_id}.jpg')
    img = cv2.imread(img_path)
    annotation =  cube_seg(img)
    box = extract_bounding_box(annotation)
    roi, x_min, y_min = extract_cube(img, box)
    annotation = cube_seg(roi, segment_everything=True)
    contours = get_square_contours(annotation)
    mid_points = [calculate_midpoint(cnt) for cnt in contours]
    clusters = find_rubik_surface(mid_points)
    ordered_clusters = [get_ordered_rubik_points(clust, shift=np.array([x_min, y_min])) for clust in clusters][1:]
    Q13d, Q23d, Q33d = get_cube_surfaces()

    # Q1 = [(167, 244), (205, 243), (241, 242), (158, 216), (196, 215), (234, 213), (147, 185), (186, 184), (225, 183)]
    # Q2 = [(147, 156),  (186, 155), (226, 153), (158, 134),(195, 132), (232, 131), (166, 114),(202, 113), (237, 111)  ]
    # Q13d, Q23d, Q33d = get_cube_surfaces()
    # success, rotation_vector, translation_vector = cv2.solvePnP(
    #     np.array(np.vstack([Q13d]), dtype=np.float32), np.array(np.vstack([Q1]), dtype=np.float32), K, dist_coeffs, flags=cv2.SOLVEPNP_IPPE)
    #
    # def reproject_points(points3d, rvec, tvec, camera_matrix):
    #     points2d, _ = cv2.projectPoints(points3d, rvec, tvec, camera_matrix, dist_coeffs)
    #     return points2d.squeeze()
    # points3dall = np.vstack([Q13d, Q23d, Q33d]).astype(np.float32)
    # # Reproject the original points
    # reprojected_points = reproject_points(points3dall, rotation_vector, translation_vector, K, )

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Plot the image and points


    # Plot enumerated points with small black circles and text
    for clust in ordered_clusters:
        for clust in get_flipped_points(clust):
            fig, ax = plt.subplots()
            ax.imshow(img_rgb)
            for i, (x, y) in enumerate(clust):
                ax.plot(x, y, 'kx', markersize=5)  # Small black 'x' mark
                ax.text(x, y, str(i), color='red', fontsize=10, ha='right', va='bottom')

            plt.title(f'Image ID: {img_id}')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.show()


    # # PLOT
    # if True:
    #     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     fig, ax = plt.subplots(figsize=(10, 10))
    #     ax.imshow(img_rgb)
    #     for point in Q1+Q2:
    #         x, y = point
    #         ax.plot(x, y, 'ro', markersize=10)  # 'ro' means red circle
    #     # Plot reprojected points
    #     for point in reprojected_points:
    #         x, y = point
    #         ax.plot(x, y, 'bo', markersize=8, label='Reprojected points')
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #     ax.set_title('Iterative ApproxPolyDP Hexagon')
    #     plt.show()

def estimate_cube_pose_que(K, dist_coeffs, cube_seg: CubeSegmentation, frame_queue, result_queue):
    tracker = cv2.TrackerCSRT.create()
    box = None
    while True:
        frame = frame_queue.get()
        if box is None:
            print('getting box slow')
            box, _ = cube_seg.detect_cube(frame)
            tracker.init(frame, (box['x'], box['y'], box['width'], box['height']))

        else:
            print('getting box fast')
            success, box = tracker.update(frame)
            if success is False:
                print('FALLBACK getting box slow')
                box, _ = cube_seg.detect_cube(frame)
            else:
                box = dict(x=box[0], y=box[1], width=box[2], height=box[3])

        if box is not None:
            rvec, tvec = estimate_cube_pose(cube_seg=cube_seg, box=box, img=frame, K=K, dist_coeffs=dist_coeffs)
            if rvec is not None:
                result_queue.put((box, tvec, rvec))
            else:
                box = None