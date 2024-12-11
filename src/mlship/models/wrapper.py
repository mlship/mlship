import numpy as np

class ModelWrapper:
    """Base wrapper to add get_model_info method to models"""
    def __init__(self, model, model_type, input_type, output_type, feature_names=None, classes=None):
        self.model = model
        self.model_type = model_type
        self.input_type = input_type
        self.output_type = output_type
        self.feature_names = feature_names
        self.classes = classes

    def predict(self, inputs):
        if self.input_type == 'numeric':
            return self.model.predict(np.array(inputs))
        return self.model.predict(inputs)

    def get_model_info(self):
        info = {
            "type": self.model_type,
            "input_type": self.input_type,
            "output_type": self.output_type,
            "features": self.feature_names,
            "n_features": len(self.feature_names) if self.feature_names else None
        }
        if self.classes is not None:
            info["classes"] = self.classes
        return info 