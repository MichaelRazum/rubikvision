import cv2
from color_classifier import ColorClassiferKmeans
from rubikvision.utils import find_webcam_index


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
    main(train=True)