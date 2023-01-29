# LSTM_forecasting
Example of time series forecasting using LSTM (Long Short-Term Memory) neural network in Python using the Keras library.

- Importing required libraries: The script starts by importing the required libraries such as pandas, numpy, scikit-learn, matplotlib and Keras.

- Data preparation: The script reads a CSV file containing the historical data of a any stock/currency/crypto, and stores it in a pandas dataframe. Then, it sorts the data based on the 'Date' column and selects the 'Open', 'High', 'Low', 'Close' and 'Volume' columns as the features to be used for predictions. The feature data is then scaled to the range of (0,1) using MinMaxScaler. Finally, the data is split into training and test sets with 80% of the data being used for training and the remaining 20% being used for testing.

- Data processing: The script defines a function 'create_dataset' to convert the feature data into a time series dataset. The function takes the feature data as an input and returns two arrays, 'dataX' and 'dataY', where 'dataX' represents the input sequence and 'dataY' represents the target. The script uses a 'look_back' argument to specify the number of time steps to be included in the input sequence, in this case, it's 7. The input data is then reshaped to fit the LSTM model.

- LSTM Model: The script creates a sequential LSTM model using Keras, with multiple LSTM layers and a single dense output layer. The model is compiled with the 'mean_squared_error' loss function and 'adam' optimizer.

- Model training: The model is fit to the training data with 100 epochs and a batch size of 32. Early stopping is used with a patience of 10 epochs, which means that if the validation loss does not improve for 10 epochs, the training will stop and the weights of the best performing epoch will be restored.

- Model evaluation: The script evaluates the model using root mean squared error (RMSE) on both the training and test sets. Finally, the script plots the training loss for the model.

- Overall, hopefully this script provides a good starting point for anyone looking to build a time series forecasting model using LSTM in Keras.
