# preprocess_module.py
from monai.transforms import ScaleIntensity, Resize

class Preprocessor:
    def __call__(self, data):
        data = ScaleIntensity()(data)
        data = Resize(spatial_size=(224,224))(data)
        return data
