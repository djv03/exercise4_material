import torch as t
import torch.nn as nn
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from model import ResNet

# load the data from the csv file and perform a train-test-split
# this can be accomplished using the already imported pandas and sklearn.model_selection modules
# TODO
df= pd.read_csv('data.csv', sep=';')
train_df, val_df= train_test_split(df, test_size=0.2, random_state=42 )

# set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
# TODO
train_dataset= ChallengeDataset(train_df,mode="train")
val_dataset= ChallengeDataset(val_df,mode="val")

BATCH_SIZE= 16

train_loader= t.utils.data.DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=2)
val_loader= t.utils.data.DataLoader(val_dataset,batch_size=BATCH_SIZE,shuffle=False,num_workers=2)

# create an instance of our ResNet model
# TODO
model = ResNet()

# set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
criterion= nn.BCEWithLogitsLoss()

# set up the optimizer (see t.optim)
optimizer= t.optim.Adam(model.parameters(),lr=0.01)

# create an object of type Trainer and set its early stopping criterion


# TODO

# go, go, go... call fit on trainer
res = #TODO

# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')