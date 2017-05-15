from .manual_feature_classifier import ManualFeatureClassifier
from keras.models import load_model


class ANNClassifier(ManualFeatureClassifier):
    def _set_model(self, data):
        return load_model(data['model'])
