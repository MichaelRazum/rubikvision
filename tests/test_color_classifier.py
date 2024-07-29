from color_classifier import ColorClassiferKmeans, ColorClassiferRoboflow, ColorClassfierEnsemble
import numpy as np

def test_classifer_kmean():
    clf = ColorClassiferKmeans().load()
    blue_image = np.array([[[255, 0, 0] for _ in range(5)] for _ in range(5)], dtype=np.uint8)
    colors = clf.estimate_colors(blue_image,[[2,2]])
    assert colors == ['blue']

def test_classfier_roboflow():
    clf = ColorClassiferRoboflow().load()
    orange_image = np.array([[[8, 146, 204] for _ in range(5)] for _ in range(5)], dtype=np.uint8)
    colors = clf.estimate_colors(orange_image,[[2,2]])
    assert colors == ['orange']

    orange_image = np.array([[[6, 25, 205] for _ in range(5)] for _ in range(5)], dtype=np.uint8)
    colors = clf.estimate_colors(orange_image,[[2,2]])
    assert colors == ['red']

def test_ensemble_classifer():
    clf = ColorClassfierEnsemble(sensitive_colors=['red', 'orange'])
    orange_image = np.array([[[8, 146, 204] for _ in range(5)] for _ in range(5)], dtype=np.uint8)
    blue_image = np.array([[[255, 0, 0] for _ in range(5)] for _ in range(5)], dtype=np.uint8)
    blue_orange_image  =np.vstack([blue_image, orange_image])
    colors = clf.estimate_colors(blue_orange_image, [[0,0], [0,9]])
    assert colors == ['blue', 'orange']