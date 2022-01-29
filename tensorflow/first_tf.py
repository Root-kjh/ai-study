import tensorflow as tf
import pandas as pd
import numpy as np
class DataSet:
    def __init__(self, x_dataset, y_dataset) -> None:
        self.__x_dataset=x_dataset
        self.__y_dataset=y_dataset

    def get_x(self):
        return self.__x_dataset

    def get_y(self):
        return self.__y_dataset

def get_datas():
    data = pd.read_csv('gpascore.csv')
    x = []
    y = data['admit'].values
    for _, rows in data.iterrows():
        x.append([rows['gre'], rows['gpa'], rows['rank']])
    return DataSet(x, y)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(128, activation='tanh'),
    tf.keras.layers.Dense(1, activation='sigmoid'),    
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

data_set = get_datas()
model.fit(x=np.array(data_set.get_x()), y=data_set.get_y(), epochs=10000)
print(model.predict([[750, 3.70, 3], [400, 2.2, 1]]))