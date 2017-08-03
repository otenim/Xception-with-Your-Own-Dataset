from keras.applications.xception import Xception, preprocess_input, decode_predictions
from keras.preprocessing import image
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    'image_file',
)

def main(args):
    image_file = os.path.abspath(args.image_file)

    model = Xception(include_top=True, weights='imagenet')
    img = image.load_img(image_file, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    preds = model.predict(preprocess_input(x))
    results = decode_predictions(preds, top=5)[0]
    for result in results:
        print(result)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
