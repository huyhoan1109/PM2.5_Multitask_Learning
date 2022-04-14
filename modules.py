import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epochs = 100
latent = None
in_seq_len = 10
out_seq_len = 6
meteo_size = 6
n_tasks = 12
lr = 0.001
lr_decay = 0.0001
w_decay = 0.0001
batch_size = 64