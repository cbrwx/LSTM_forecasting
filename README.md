# LSTM_forecasting
Example of time series forecasting using LSTM (Long Short-Term Memory) neural network in Python using the Keras library.

This code uses the Keras library to build and train an LSTM (Long-Short-Term-Memory) model for time series prediction. The code first reads in the data from a csv file and preprocesses the data by sorting it by the "Date" column and scaling the feature data to be in the range (0, 1). The code then splits the preprocessed data into training and test sets, converts the feature data into a time series dataset, reshapes the data for the LSTM model, and creates the LSTM model using sequential API of Keras. The model has three LSTM layers with 150, 150 and 100 neurons respectively, and a dense output layer with 1 neuron. The model is compiled using the "mean_squared_error" loss and the "adam" optimizer. The model is then fit to the training data with a batch size of 6 and a maximum of 100 epochs. The model training is early stopped if the validation loss does not decrease for 10 epochs. Finally, the model performance is evaluated on the training and test sets by computing the root mean squared error (RMSE) between the true and predicted values. The training loss is also plotted over the epochs to visualize the model's convergence.

The code can be used for time series prediction tasks where the goal is to predict a future value based on a sequence of past values. The code can be modified for different time series datasets and for different prediction goals by changing the preprocessing steps, the model architecture, the loss function and the optimizer.

- Importing required libraries: The script starts by importing the required libraries such as pandas, numpy, scikit-learn, matplotlib and Keras.

- Data preparation: The script reads a CSV file containing the historical data of a any stock/currency/crypto, and stores it in a pandas dataframe. Then, it sorts the data based on the 'Date' column and selects the 'Open', 'High', 'Low', 'Close' and 'Volume' columns as the features to be used for predictions. The feature data is then scaled to the range of (0,1) using MinMaxScaler. Finally, the data is split into training and test sets with 80% of the data being used for training and the remaining 20% being used for testing.

- Data processing: The script defines a function 'create_dataset' to convert the feature data into a time series dataset. The function takes the feature data as an input and returns two arrays, 'dataX' and 'dataY', where 'dataX' represents the input sequence and 'dataY' represents the target. The script uses a 'look_back' argument to specify the number of time steps to be included in the input sequence, in this case, it's 7. The input data is then reshaped to fit the LSTM model.

- LSTM Model: The script creates a sequential LSTM model using Keras, with multiple LSTM layers and a single dense output layer. The model is compiled with the 'mean_squared_error' loss function and 'adam' optimizer.

- Model training: The model is fit to the training data with 100 epochs and a batch size of 32. Early stopping is used with a patience of 10 epochs, which means that if the validation loss does not improve for 10 epochs, the training will stop and the weights of the best performing epoch will be restored.

- Model evaluation: The script evaluates the model using root mean squared error (RMSE) on both the training and test sets. Finally, the script plots the training loss for the model.

- The line "early_stopping = EarlyStopping(patience=10, restore_best_weights=True)" sets up early stopping, and the callback "early_stopping" is passed to the "fit" method of the model, so that it will be used during training. 

- Overall, hopefully this script provides a good starting point for anyone looking to build a time series forecasting model using LSTM in Keras.
