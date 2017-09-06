from keras.applications.xception import Xception
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.models import Model, load_model
import numpy as np
import os
import argparse

current_directory = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument(
    'model',
)
parser.add_argument(
    'image',
)
parser.add_argument(
    '--top',
    type=int,
    default=10,
)
parser.add_argument(
    '--classes',
    default=os.path.join(current_directory, 'classes.txt'),
)

def main(args):

    # create model
    model = load_model(args.model)

    # create classes
    classes = []
    with open(args.classes, 'r') as f:
        classes = list(map(lambda x: x.strip(), f.readlines()))

    # load image
    img = image.load_img(args.image, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # predict
    preds = model.predict(x)
    pred = preds[0]
    result = [(classes[i], float(pred[i]) * 100.0) for i in range(len(pred))]
    result.sort(reverse=True, key=lambda x: x[1])
    for i in range(args.top):
        (class_name, prob) = result[i]
        print("Top %d ====================" % (i + 1))
        print("Class name: %s" % (class_name))
        print("Probability: %.2f%%" % (prob))

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
