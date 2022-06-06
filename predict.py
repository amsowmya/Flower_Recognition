from keras.models import load_model
from keras.preprocessing import image
import numpy as np

class flower:
    def __init__(self, fileName):
        self.fileName = fileName

    def predict_flower(self):
        model = load_model('saved_models/model.h5')

        test_image = image.load_img(self.fileName, target_size=(64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)

        result = model.predict(test_image)
        print(result)

        if result[0][0] == 1:
            predictor = 'daisy'
            return [{'image': predictor}]
        elif result[0][1] == 1:
            predictor = 'dandelion'
            return [{'image': predictor}]
        elif result[0][2] == 1:
            predictor = 'rose'
            return [{'image': predictor}]
        elif result[0][3] == 1:
            predictor = 'sunflower'
            return [{'image': predictor}]
        else:
            predictor = 'tulip'
            return [{'image': predictor}]

# if result[0][0] == 1:
#     predictor = 'daisy'
#     print(predictor)
# elif result[0][1] == 1:
#     predictor = 'dandelion'
#     print(predictor)
# elif result[0][2] == 1:
#     predictor = 'rose'
#     print(predictor)
# elif result[0][3] == 1:
#     predictor = 'sunflower'
#     print(predictor)
# else:
#     predictor = 'tulip'
#     print(predictor)

