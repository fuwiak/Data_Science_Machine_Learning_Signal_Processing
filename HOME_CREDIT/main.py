import pandas as pd
import numpy as np
import matplotlib as plt
from contextlib import contextmanager

url = "../home_credit_source/application_train.csv"

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))




dataset=pd.read_csv(url)

#description_set = data.describe()

def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

def application_train_test(num_rows = None, nan_as_category = True):
    # Read data and merge
    df = pd.read_csv(url, nrows= num_rows)
    test_df = pd.read_csv(url, nrows= num_rows)
    print("Train samples: {}, test samples: {}".format(len(df), len(test_df)))
    df = df.append(test_df).reset_index()


