import abc
import os
import pickle
import numpy as np
import cv2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from roboflow import Roboflow

class ColorClassifer(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def estimate_colors(self, img, points):
        raise NotImplementedError()

    def load(self):
        return self


class ColorClassiferRoboflow(ColorClassifer):
    def __init__(self, RADIUS=4):
        api_key = os.environ.get('ROBO_KEY')
        assert api_key is not None, 'enter api key here'
        rf = Roboflow(api_key=api_key)
        project = rf.workspace().project("colour-detection")
        model = project.version(5).model
        self.model = model
        self.RADIUS = RADIUS

    def predict(self, data: np):
        result = self.model.predict(data,).json()
        return result['predictions'][0]['class']

    def estimate_colors(self, img, points):
        max_y, max_x, _ = img.shape
        predictions = []
        for p in points:
            box_min_y = max(0, p[1] - self.RADIUS)
            box_max_y = min(max_y, p[1] + self.RADIUS)
            box_min_x = max(0, p[0] - self.RADIUS)
            box_max_x = min(max_x, p[0] + self.RADIUS)
            prediction = self.predict(img[box_min_y:box_max_y,
                                          box_min_x:box_max_x,])
            predictions.append(prediction)
        return predictions


    def load(self):
        return ColorClassiferRoboflow()


class ColorClassiferKmeans(ColorClassifer):
    def __init__(self, n_neighbors=3):
        self.le = None
        self.clf = None
        self.n_neighbors = n_neighbors

    def bgr2clf_format(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    def train_classifier(self, samples):
        X = []
        y = []
        for color, color_samples in samples.items():
            X.extend(color_samples)
            y.extend([color] * len(color_samples))

        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        clf = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        clf.fit(X, y_encoded)
        self.clf = clf
        self.le = le

    def save_classifier(self):
        model_path = os.path.join(os.path.dirname(__file__), 'weights', 'color_classifier.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump((self.clf, self.le), f)

    def load(self, name='color_classifier.pkl'):
        model_path = os.path.join(os.path.dirname(__file__), 'weights', name)
        with open(model_path, 'rb') as f:
            clf, le = pickle.load(f)
            self.clf = clf
            self.le = le
        return self

    def predict(self, data):
        color_index = self.clf.predict([data[1:]])[0]
        prediction = self.le.inverse_transform([color_index])[0]
        return prediction

    def estimate_colors(self, img, points, radius=1, threshold=.65):
        img_lab = self.bgr2clf_format(img)
        h, w = img_lab.shape[:2]
        predictions = []
        
        for p in points:
            x, y = int(p[0]), int(p[1])
            
            # Sample surrounding region
            samples = []
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        samples.append(img_lab[ny, nx])
            
            sample_predictions = [self.predict(sample) for sample in samples]
            colors, counts = np.unique(sample_predictions, return_counts=True)
            N = len(sample_predictions)
            n = np.max(counts)
            if n / N < threshold:
                raise Exception('unclear prediction')
            majority_color = colors[np.argmax(counts)]
            predictions.append(majority_color)

        return predictions

class ColorClassfierEnsemble(ColorClassifer):
    def __init__(self, sensitive_colors:list):
        self.clf = ColorClassiferKmeans().load()
        self.clf_robo = ColorClassiferRoboflow()
        assert sensitive_colors != [], 'enter sensitive colors where roboflow is triggered'
        self.sensitive_colors = sensitive_colors

    def estimate_colors(self, img, points):
        colors = self.clf.estimate_colors(img, points)
        for n, col in enumerate(colors):
            if col in self.sensitive_colors:
                col_rob = self.clf_robo.estimate_colors(img, [points[n]])
                if not col_rob[0] == col:
                    print(f'got {col_rob[0]=} {col=} for point {points[n]}')
                    raise  Exception('Wrong color')
        return colors
