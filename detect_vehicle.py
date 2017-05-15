from moviepy.editor import VideoFileClip
from utils.svm_classifier import SVMClassifier
from utils.cnn_classifier import CNNClassifier
from utils.ann_classifier import ANNClassifier
from scipy.ndimage.measurements import label

import numpy as np
import pickle
import cv2
import sys


scales = np.linspace(0.5, 2.25, 8)
pix_per_step = 16
ystart, ystop = 400, 656
denoise_th = 1
block_th = 2


def load_classifier(model_file):
    ext = model_file.split('.')[-1]
    if ext == 'cnn':
        # Keras CNN model here
        with open(model_file, 'rb') as f:
            data = pickle.load(f)
        clf = CNNClassifier(data)
    elif ext == 'svm':
        # Scikit-learn SVM model here
        with open(model_file, 'rb') as f:
            data = pickle.load(f)
        clf = SVMClassifier(data)
    elif ext == 'ann':
        # Keras ANN model here
        with open(model_file, 'rb') as f:
            data = pickle.load(f)
        clf = ANNClassifier(data)
    else:
        raise ValueError('Model file must be either .cnn or .svm file')

    return clf


def accum_heatmap(heatmap, response_map, window_width):
    out_map = np.copy(heatmap)
    for x, y in zip(*response_map.nonzero()):
        out_map[x:x+window_width, y:y+window_width] += 1
    return out_map


def extract_bboxes(heatmap, denoise_th, block_th):
    bboxes = []
    hmap = np.copy(heatmap)
    hmap[hmap <= denoise_th] = 0
    denoise_label_map, denoise_label_count = label(hmap)
    for i in range(1, denoise_label_count+1):
        hmap[denoise_label_map == i] = np.max(hmap[denoise_label_map == i])
    hmap[hmap <= block_th] = 0
    bboxes_label_map, bboxes_label_count = label(hmap)
    for i in range(1, bboxes_label_count+1):
        nonzero = (bboxes_label_map == i).nonzero()
        min_x = np.min(nonzero[1])
        max_x = np.max(nonzero[1])
        min_y = np.min(nonzero[0])
        max_y = np.max(nonzero[0])
        bboxes.append(((min_x, min_y), (max_x, max_y)))
    return bboxes


def draw_bboxes(image, bboxes, color=(0, 0, 255), thick=6):
    draw_img = np.copy(image)
    for bbox in bboxes:
        cv2.rectangle(draw_img, bbox[0], bbox[1], color=color, thickness=thick)
    return draw_img


def process_image(image):
    global scales, pix_per_step
    global ystart, ystop
    heatmap = np.zeros(image.shape[:2])
    for scale in scales:
        response_map = clf.predict_all(image, ystart, ystop, scale, pix_per_step)
        heatmap += response_map
    bboxes = extract_bboxes(heatmap, denoise_th, block_th)
    draw_img = draw_bboxes(image, bboxes)
    return draw_img


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: detect_vehicle model_file input_video output_video")
        sys.exit(0)

    input_video = sys.argv[2]
    output_video = sys.argv[3]
    clf = load_classifier(sys.argv[1])

    clip = VideoFileClip(input_video)
    write_clip = clip.fl_image(process_image)
    write_clip.write_videofile(output_video, audio=False)
