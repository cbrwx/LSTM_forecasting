import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping

plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = 'black'
plt.rcParams['axes.facecolor'] = 'black'
plt.rcParams['axes.edgecolor'] = 'white'
plt.rcParams['grid.color'] = 'white'
plt.rcParams['text.color'] = 'white'

from keras.models import Sequential
from keras.layers import Dense, LSTM

# Read the CSV file, using data from yahoo, so it needs to feature data into a time series dataset
df = pd.read_csv('http://lillegaardtannklinikk.no/1/BTC-USD.csv', header=0, parse_dates=['Date'])

# Sort the data by the 'Date' column
df = df.sort_values(by='Date')

# Select the 'Open', 'High', 'Low', 'Close', 'Volume' columns as the features to be predicted
features = df[['Open', 'High', 'Low', 'Close', 'Volume']]

# Scale the feature data to be in the range (0, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
features = scaler.fit_transform(features.values)

# Split the data into training and test sets
train_size = int((len(features) * 0.8))
train_features, test_features = features[0:train_size], features[train_size:]

# Convert the feature data into a time series dataset
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 3])
    return np.array(dataX), np.array(dataY)

X_train, y_train = create_dataset(train_features, 7)
X_test, y_test = create_dataset(test_features, 7)

# Reshape the data for LSTM model
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 5))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 5))

# Create an LSTM model
model = Sequential()
model.add(LSTM(150, input_shape=(X_train.shape[1], 5), return_sequences=True))
model.add(LSTM(150, return_sequences=True))
model.add(LSTM(100, return_sequences=False))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

#Fit the model to the training data
early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=100, batch_size=6, validation_data=(X_test, y_test), callbacks=[early_stopping])

#Evaluate the model
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

train_rmse = np.sqrt(mean_squared_error(y_train, train_predict))
test_rmse = np.sqrt(mean_squared_error(y_test, test_predict))

print(f"Train RMSE: {train_rmse:.2f}")
print(f"Test RMSE: {test_rmse:.2f}")

#Plot the training loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('LSTM Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()
