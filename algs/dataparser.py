import os
import cv2

def load_video(filepath):
    cap = cv2.VideoCapture(filepath)
    data = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            data.append(frame)
    return data


def load_labels(filepath):
    file = open(filepath)
    lines = [float(line.strip()) for line in file]
    return lines


def load_training_data():
    return load_video("../data/train.mp4"), load_labels("../data/train.txt")


def load_testing_data():
    return load_video("../data/test.mp4")


def save_data(frames, labels, foldername):
    base = os.path.join("../data/formatted/", foldername)
    save_labels(labels, os.path.join(base, "labels.txt"))
    save_frames(frames, os.path.join(base))


def save_frames(frames, folderpath):
    for i, frame in enumerate(frames):
        cv2.imwrite(folderpath + "{}.png".format(str(i)), frame)


def save_labels(labels, filepath):
    output = '\n'.join([str(label) for label in labels])
    file = open(filepath, "w+")
    file.write(output.strip())
    file.close()


def split_nth_data(n):
    frames, labels = load_training_data()
    print(len(frames), len(labels))
    frames = frames[::n]
    labels = labels[::n]
    print(len(frames), len(labels))
    save_data(frames, labels, "ordered_{}".format(str(len(frames))))


split_nth_data(10)
