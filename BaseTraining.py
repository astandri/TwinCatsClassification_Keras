import pandas as pd

from KerasModel import Model

train = pd.read_csv('Dataset/train.csv')
validation = pd.read_csv('Dataset/validation.csv')

X_train, y_train = train.drop('label',axis=1), train['label']
X_val, y_val = validation.drop('label',axis=1), validation['label']

#create new instance
model = Model(X_train, y_train)

#build model
model.build_model()

#train model with base hyperparameters (1 layer, 10 nodes)
model.train_model(val_predictors=X_val, val_targets=y_val)