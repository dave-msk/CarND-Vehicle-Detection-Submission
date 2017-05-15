from abc import abstractmethod
from .classifier import Classifier
from .utils import convert_color
from .feature_extraction import *
import numpy as np


class ManualFeatureClassifier(Classifier):

    @abstractmethod
    def _set_model(self, data):
        pass

    def _set_attributes(self, data):
        self.color_space = data['color_space']
        self.orient = data['orient']
        self.pix_per_cell = data['pix_per_cell']
        self.cell_per_block = data['cell_per_block']
        self.hog_channel = data['hog_channel']
        self.spatial_size = data['spatial_size']
        self.hist_bins = data['hist_bins']
        self.spatial_feat = data['spatial_feat']
        self.hist_feat = data['hist_feat']
        self.hog_feat = data['hog_feat']
        self.window_side = data['window_side']
        self.clf = self._set_model(data)
        self.scaler = data['scaler']

    def predict_all(self, image, ystart, ystop, scale, pix_per_step):
        """
        Performs a sliding window search for vehicles
        
        :param image: input image
        :param ystart: start of y-coordinate of search area
        :param ystop: end of y-coordinate of search area
        :param scale: scale of search window relative to the model-accepted size
        :param pix_per_step: window search step size (in pixels)
        :return: binary map with positive detection turned on
        """
        img_tosearch = image[ystart:ystop, :, :]
        ctrans_tosearch = convert_color(img_tosearch, self.color_space)

        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

        response_map = np.zeros(ctrans_tosearch.shape[:2])

        # Define blocks and steps
        nxblocks = (ctrans_tosearch.shape[1] // self.pix_per_cell) - self.cell_per_block + 1
        nyblocks = (ctrans_tosearch.shape[0] // self.pix_per_cell) - self.cell_per_block + 1
        nfeat_per_block = self.orient*self.cell_per_block**2
        nblocks_per_window = (self.window_side // self.pix_per_cell) - self.cell_per_block + 1
        cells_per_step = pix_per_step // self.pix_per_cell
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
        nysteps = 4

        # Compute individual channel HOG features
        hog_feat_map = self._get_hog_features(ctrans_tosearch)

        # Extract features for each sub-image
        test_features = []
        coords = []
        for xb in range(nxsteps):
            xpos = xb*cells_per_step
            xleft = xb*pix_per_step
            for yb in range(nysteps):
                ypos = yb*cells_per_step
                ytop = yb*pix_per_step
                feature = []

                sub_img = ctrans_tosearch[ytop:ytop + self.window_side, xleft:xleft + self.window_side]

                # Extract spatial binned features if applicable
                if self.spatial_feat:
                    spatial_features = bin_spatial(sub_img, self.spatial_size)
                    feature.append(spatial_features)

                # Extract color histogram features if applicable
                if self.hist_feat:
                    hist_features = color_hist(sub_img, nbins=self.hist_bins)
                    feature.append(hist_features)

                # Extract HOG for this patch
                if self.hog_feat:
                    if self.color_space != 'GRAY' and self.hog_channel == 'ALL':
                        hog_feat1 = hog_feat_map[0][ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                        hog_feat2 = hog_feat_map[1][ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                        hog_feat3 = hog_feat_map[2][ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                        hog_feature = np.hstack((hog_feat1, hog_feat2, hog_feat3))
                    else:
                        hog_feature = hog_feat_map[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                    feature.append(hog_feature)

                # Record extracted features
                feature = np.concatenate(feature)
                test_features.append(feature)
                coords.append((xleft, ytop))

        # Perform prediction
        test_features = np.array(test_features)
        scaled_test_features = self.scaler.transform(np.array(test_features))
        preds = self.clf.predict(scaled_test_features)
        preds = np.array(preds).ravel()

        # Record positive detections on a response map
        for i in range(len(preds)):
            if preds[i] >= 0.5:
                ytop = coords[i][1]
                xleft = coords[i][0]
                response_map[ytop:ytop+self.window_side, xleft:xleft+self.window_side] = 1

        # Construct the heatmap from the response map
        heatmap = np.zeros(image.shape[:2])
        heatmap[ystart:ystop, :] = cv2.resize(response_map, img_tosearch.shape[:2][::-1])
        return heatmap

    def _get_hog_features(self, img):
        if self.color_space == 'GRAY':
            return get_hog_features(img, self.orient,
                                    self.pix_per_cell,
                                    self.cell_per_block,
                                    vis=False,
                                    feature_vec=False)
        if self.hog_channel == 'ALL':
            ch1 = img[:, :, 0]
            ch2 = img[:, :, 1]
            ch3 = img[:, :, 2]
            ch1_hog = get_hog_features(ch1, self.orient,
                                       self.pix_per_cell,
                                       self.cell_per_block,
                                       vis=False,
                                       feature_vec=False)
            ch2_hog = get_hog_features(ch2, self.orient,
                                       self.pix_per_cell,
                                       self.cell_per_block,
                                       vis=False,
                                       feature_vec=False)
            ch3_hog = get_hog_features(ch3, self.orient,
                                       self.pix_per_cell,
                                       self.cell_per_block,
                                       vis=False,
                                       feature_vec=False)
            return ch1_hog, ch2_hog, ch3_hog

        return get_hog_features(img[:, :, self.hog_channel], self.orient,
                                self.pix_per_cell,
                                self.cell_per_block,
                                vis=False,
                                feature_vec=False)