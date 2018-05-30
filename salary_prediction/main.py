# -*- coding: utf-8 -*-
"""

@author: pawel
"""

import tensorflow as tf
import tensorflow.contrib as contrib
import pandas as pd
import tempfile
import numpy as np
import tempfile

train_file = "adult-training.csv"
test_file = "adult-test.csv"



COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num",
           "marital_status", "occupation", "relationship", "race", 
           "gender", "capital_gain", "capital_loss", "hours_per_week", 
           "native_country","income_bracket"]
           
LABEL_COLUMN = "label"

CATEGORICAL_COLUMNS = ["workclass", "education", "marital_status", 
                       "occupation","relationship", "race", "gender", 
                       "native_country"]
                       
CONTINUOUS_COLUMNS = ["age", "education_num", "capital_gain", 
                      "capital_loss","hours_per_week"]