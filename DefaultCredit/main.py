import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats

from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, roc_curve



file_name = "credit_clients.xls"

data = pd.ExcelFile(file_name)

df = data.parse('Data')

