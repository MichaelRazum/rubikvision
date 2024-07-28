import abc
import os
import numpy as np
import cv2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

from utils import find_webcam_index


def get_cap():
    video_stream = find_webcam_index("C922 Pro Stream Webcam")
    cap = cv2.VideoCapture(video_stream)
    return cap


def sample_colors(colors, N=10):
    cap = get_cap()
    samples = {color: [] for color in colors}
    current_color_index = 0

    def mouse_callback(event, x, y, flags, param):
        nonlocal current_color_index
        if event == cv2.EVENT_LBUTTONDOWN:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[y, x]
            samples[colors[current_color_index]].append(lab[1:])
            if len(samples[colors[current_color_index]]) == N:
                current_color_index += 1

    cv2.namedWindow("Color Sampler")
    cv2.setMouseCallback("Color Sampler", mouse_callback)

    while current_color_index < len(colors):
        ret, frame = cap.read()
        if not ret:
            break

        cv2.putText(frame, f"Click on {colors[current_color_index]} ({len(samples[colors[current_color_index]])}/{N})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Color Sampler", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return samples



def predict_color(col_clf):
    cap = get_cap()
    prediction = "No prediction yet"

    def mouse_callback(event, x, y, flags, param):
        nonlocal prediction
        if event == cv2.EVENT_LBUTTONDOWN:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[y, x]
            prediction = col_clf.predict(lab)

    cv2.namedWindow("Color Prediction")
    cv2.setMouseCallback("Color Prediction", mouse_callback)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.putText(frame, f"Predicted Color: {prediction}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Click to predict color, 'q' to quit", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Color Prediction", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

class ColorClassifer(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def predict(self, data: np):
        raise NotImplementedError()

    @abc.abstractmethod
    def load(self):
        raise NotImplementedError()

    def bgr2clf_format(self, img):
        return img


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


    def load(self):
        model_path = os.path.join(os.path.dirname(__file__), 'weights', 'color_classifier.pkl')
        with open(model_path, 'rb') as f:
            clf, le = pickle.load(f)
            self.clf = clf
            self.le = le
        return self

    def predict(self, data):
        color_index = self.clf.predict([data[1:]])[0]
        prediction = self.le.inverse_transform([color_index])[0]
        return prediction

def main(train):

    col_clf = ColorClassiferKmeans()
    if train:
        colors = ['red', 'orange', 'green', 'white', 'blue', 'yellow']
        N = 10
        print("Starting color sampling...")
        samples = sample_colors(colors, N)
        col_clf.train_classifier(samples)
        col_clf.save_classifier()

    col_clf.load()
    predict_color(col_clf)


if __name__ == "__main__":
    main(train=False)