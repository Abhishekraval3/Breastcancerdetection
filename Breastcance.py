import pandas as pd
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

from keras.models import Sequential
from keras.layers import Dense
     

data = load_breast_cancer()

data.keys()
     

print(data['DESCR'])
     

data['data'].shape
     

data['feature_names']
     

data['data'][0]
     

j = 0
for i in data['feature_names']:
  print(i,":",data['data'][0][j])
  j+=1
     

feature = data['data']
     

label = data['target']

data['target_names']
     

feature.shape
     

label.shape
     

scale = StandardScaler()

feature = scale.fit_transform(feature)
     

j = 0
for i in data['feature_names']:
  print(i,":",feature[0][j])
  j+=1
     

print(feature[568])
print(data['target_names'][label[568]],label[568])

df_frt = pd.DataFrame(feature , columns = data['feature_names'])
df_lbl = pd.DataFrame(label , columns = ['label'])
df = pd.concat([df_frt, df_lbl], axis=1)
df = df.sample(frac = 1)

feature = df.values[ : , : 30]
label = df.values[ : ,30: ]
     

df
     

#500 Training
X_train = feature[:500]
y_train = label[:500]

#35 Validation
X_val = feature[500:535]
y_val = label[500:535]

#34 Testing
X_test = feature[535:]
y_test = label[535:]

model = Sequential()

model.add(Dense(32, activation = 'relu', input_dim = 30))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.compile( loss = 'binary_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'])
     

model.fit( X_train , y_train, epochs = 10, batch_size = 5, validation_data = (X_val, y_val))
     

model.evaluate(X_test , y_test)
     

model.evaluate(X_val , y_val)

for i in range(30):
  sample = X_test[i]
  sample = np.reshape(sample, (1,30))

  if (model.predict(sample)[0][0] > 0.5):
    print("-Benign")
  else:
    print("-Malignant")

  if (y_test[i] == 1):
    print("*Banign")
  else:
    print("*Melignant")
  print("-----------")
     

t = 0
for i in y_val:
  if (i == 1):
    t += 1

print(t)

t = 0
for i in y_test:
  if (i == 1):
    t += 1

print(t)
     

X_test[0] * -.1