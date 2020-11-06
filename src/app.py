import math
import os

import albumentations
import numpy as np
import torch
from flask import Flask, render_template, request
from wtfml.data_loaders.image.classification import ClassificationDataset
from wtfml.engine import Engine

from main import CV_Model

UPLOAD = "static/image_folder/"
MODEL_NAME = 'se_resnext50_32x4d'
UPLOAD = "static/image_folder/"
DEVICE = "cpu"
MODEL = None
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
MODEL_PATH = "../model/"

app = Flask(__name__)


def predict(image_path, model):
    '''
    Make prediction using trained model
    '''

    test_aug = albumentations.Compose(
        [
            albumentations.Normalize(
                            mean=MEAN,
                            std=STD,
                            max_pixel_value=255.0,
                            always_apply=True)
        ]
    )

    test_images = [image_path]
    test_targets = [0]

    test_dataset = ClassificationDataset(
        image_paths=test_images,
        targets=test_targets,
        resize=None,
        augmentations=test_aug
        )

    test_loader = torch.utils.data.DataLoader(
                        dataset=test_dataset,
                        batch_size=1,
                        shuffle=False,
                        num_workers=0)

    engine = Engine(model=model, optimizer=None, device=DEVICE)
    predictions = engine.predict(data_loader=test_loader)
    return np.vstack((predictions)).reshape(-1)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


@app.route("/", methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        file = request.files["image"]
        image_location = os.path.join(UPLOAD, file.filename)
        file.save(image_location)
        pred = predict(image_location, MODEL)
        pred = round(sigmoid(pred)*100, 2)
        return render_template("index.html",
                               image_prediction=pred,
                               image_location=image_location,
                               filename=file.filename)
    return render_template("index.html",
                           image_prediction="",
                           image_location=None,
                           filename=None)


if __name__ == "__main__":
    MODEL = CV_Model(pretrained="imagenet")
    MODEL.load_state_dict(torch.load(os.path.join(MODEL_PATH, "model.bin")))
    MODEL.to(DEVICE)
    MODEL.eval()
    app.run(port=8888, debug=True)
