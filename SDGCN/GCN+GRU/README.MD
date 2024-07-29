This is a TensorFlow implementation of T-GCN: A Temporal Graph Convolutional Network for Traffic Prediction.

## Requirements:
* tensorflow
* scipy
* numpy
* matplotlib
* pandas
* math

## Run the demo
Python main.py

Our baselines included: <br>
(1) History Average model (HA)<br>
(2) Autoregressive Integrated Moving Average model (ARIMA)<br>
(3) Support Vector Regression model (SVR)<br>
(4) Graph Convolutional Network model (GCN)<br>
(5) Gated Recurrent Unit model (GRU)<br>

The python implementations of HA/ARIMA/SVR models were in the baselines.py; The GCN and GRU models were in gcn.py and gru.py respective.

