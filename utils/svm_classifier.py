from .manual_feature_classifier import ManualFeatureClassifier


class SVMClassifier(ManualFeatureClassifier):
    def _set_model(self, data):
        return data['clf']