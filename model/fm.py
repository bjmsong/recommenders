import sys

sys.path.append("../../")
import os
import papermill as pm
from tempfile import TemporaryDirectory
import xlearn as xl
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from reco_utils.common.constants import SEED
from reco_utils.common.timer import Timer
from reco_utils.recommender.deeprec.deeprec_utils import (
    download_deeprec_resources, prepare_hparams
)
from reco_utils.recommender.deeprec.models.xDeepFM import XDeepFMModel
from reco_utils.recommender.deeprec.io.iterator import FFMTextIterator
from reco_utils.tuning.parameter_sweep import generate_param_grid
from reco_utils.dataset.pandas_df_utils import LibffmConverter

print("System version: {}".format(sys.version))
print("Xlearn version: {}".format(xl.__version__))

# Parameters
YAML_FILE_NAME = "xDeepFM.yaml"
TRAIN_FILE_NAME = "cretio_tiny_train"
VALID_FILE_NAME = "cretio_tiny_valid"
TEST_FILE_NAME = "cretio_tiny_test"
MODEL_FILE_NAME = "model.out"
OUTPUT_FILE_NAME = "output.txt"

LEARNING_RATE = 0.2
LAMBDA = 0.002
# The metrics for binary classification options are "acc", "prec", "f1" and "auc"
# for regression, options are "rmse", "mae", "mape"
METRIC = "auc"
EPOCH = 10
OPT_METHOD = "sgd"  # options are "sgd", "adagrad" and "ftrl"

if __name__ == "__main__":
    df_feature_original = pd.DataFrame({
        'rating': [1, 0, 0, 1, 1],
        'field1': ['xxx1', 'xxx2', 'xxx4', 'xxx4', 'xxx4'],
        'field2': [3, 4, 5, 6, 7],
        'field3': [1.0, 2.0, 3.0, 4.0, 5.0],
        'field4': ['1', '2', '3', '4', '5']
    })

    converter = LibffmConverter().fit(df_feature_original, col_rating='rating')
    df_out = converter.transform(df_feature_original)

    print('There are in total {0} fields and {1} features.'.format(converter.field_count, converter.feature_count))

    tmpdir = TemporaryDirectory()

    data_path = tmpdir.name
    yaml_file = os.path.join(data_path, YAML_FILE_NAME)
    train_file = os.path.join(data_path, TRAIN_FILE_NAME)
    valid_file = os.path.join(data_path, VALID_FILE_NAME)
    test_file = os.path.join(data_path, TEST_FILE_NAME)
    model_file = os.path.join(data_path, MODEL_FILE_NAME)
    output_file = os.path.join(data_path, OUTPUT_FILE_NAME)

    if not os.path.exists(yaml_file):
        download_deeprec_resources(r'https://recodatasets.blob.core.windows.net/deeprec/', data_path,
                                   'xdeepfmresources.zip')

    # Training task
    ffm_model = xl.create_ffm()  # Use field-aware factorization machine (ffm)
    ffm_model.setTrain(train_file)  # Set the path of training dataset
    ffm_model.setValidate(valid_file)  # Set the path of validation dataset

    # Parameters:
    #  0. task: binary classification
    #  1. learning rate: 0.2
    #  2. regular lambda: 0.002
    #  3. evaluation metric: auc
    #  4. number of epochs: 10
    #  5. optimization method: sgd
    param = {"task": "binary",
             "lr": LEARNING_RATE,
             "lambda": LAMBDA,
             "metric": METRIC,
             "epoch": EPOCH,
             "opt": OPT_METHOD
             }

    # Start to train
    # The trained model will be stored in model.out
    with Timer() as time_train:
        ffm_model.fit(param, model_file)

    # Prediction task
    ffm_model.setTest(test_file)  # Set the path of test dataset
    ffm_model.setSigmoid()  # Convert output to 0-1

    # Start to predict
    # The output result will be stored in output.txt
    with Timer() as time_predict:
        ffm_model.predict(model_file, output_file)

    with open(output_file) as f:
        predictions = f.readlines()

    with open(test_file) as f:
        truths = f.readlines()

    truths = np.array([float(truth.split(' ')[0]) for truth in truths])
    predictions = np.array([float(prediction.strip('')) for prediction in predictions])

    auc_score = roc_auc_score(truths, predictions)
