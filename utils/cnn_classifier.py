from .classifier import Classifier
from keras.models import load_model
from .utils import get_sliding_windows

import numpy as np
import cv2


class CNNClassifier(Classifier):
    def _set_attributes(self, data):
        self.clf = load_model(data['model'])
        self.input_size = data['input_size']

    def predict_all(self, image, ystart, ystop, scale, pix_per_step):

        img_tosearch = image[ystart:ystop, :, :]

        scaled_img_tosearch = np.copy(img_tosearch)
        if scale != 1:
            imshape = img_tosearch.shape
            scaled_img_tosearch = cv2.resize(scaled_img_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

        windows = get_sliding_windows((0, scaled_img_tosearch.shape[1]), (0, self.input_size[1]+4*pix_per_step),
                                      self.input_size[:2], (pix_per_step, pix_per_step))
        response_map = np.zeros(scaled_img_tosearch.shape[:2])
        inputs = []
        for tl, br in windows:
            inputs.append(scaled_img_tosearch[tl[1]:br[1], tl[0]:br[0], :])

        inputs = np.array(inputs)
        # print(inputs.shape)
        preds = np.int_(self.clf.predict(inputs) > 0.9)
        for i in range(len(preds)):
            if preds[i]:
                tl, br = windows[i]
                response_map[tl[1]:br[1], tl[0]:br[0]] = 1

        heatmap = np.zeros(image.shape[:2])
        heatmap[ystart:ystop, :] = cv2.resize(response_map, img_tosearch.shape[:2][::-1])

        return heatmap
