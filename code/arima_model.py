import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

df = pd.read_csv("tsf-distribution-shift/data/synthetic/synthetic_shift_v1.csv")