## Extract roads from satellite images

For this problem, a set of satellite/aerial images acquired from Google Maps and the corresponding ground-truth images are provided in  `/training`. Our goal is to train a classifier to segment roads for images in `/test_set_image`. Our convolutional neural network is provided in `run.py` and our results using the CNN are in `predictions_testing`.

This repository contains following parts:
1. Training data — download `training` from  https://inclass.kaggle.com/c/epfml-segmentation/data and put `training` in `/`
2. Testing data — download `test_set_image` from  https://inclass.kaggle.com/c/epfml-segmentation/data and put `test_set_image` in `/`
3. CNN code for road segmentation — run.py
4. A pre-computed CNN model — mnist
5. Prediction data — `predictions_testing`
6. A submission file — tf_submission.csv
7. An image generated in the code for converting the format — sharpen.png

To train your own model, please set RESTORE_MODEL = False. The training might take several hours.

To reproduce the result shown in this file, please set RESTORE_MODEL = True.

This is the work of Fan Zhang, Wenyuan Lv, and Tina Fang.