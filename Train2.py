# ==============================================================================
#
#       I insert the train section in this code
#       Ahad Aghapour Ozyegin University Department of Computer Science
#       ahad.aghapour@ozu.edu.tr
#
# ==============================================================================
import os
import pickle
import numpy as np
import pandas as pd
from sklearn import model_selection, preprocessing
from sklearn.linear_model import LogisticRegression

df = pd.read_excel(r"diabetes.xlsx")
# slice the data
data = df.values[:,0:7].astype(float)
target = df.values[:,-1].astype(int)

# LogisticRegression
logisticReg = LogisticRegression()
logisticReg.fit(data, target)

# Save the best Pickle file
pickle._dump(logisticReg, open('logisticReg.pkl', 'wb'))
print('\nyour pickle file save to: ' + os.getcwd())
