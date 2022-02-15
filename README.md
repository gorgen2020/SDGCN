This codes includes our works on Traffic Flow Prediction by Semantics-aware Dynamic Graph Convolutional Network for Traffic Flow Forecasting.

##Semantics-aware Dynamic Graph Convolutional Network for Traffic Flow Forecasting
Traffic flow forecasting is a challenging task due to the spatio-temporal dependence and the stochastic nature of the complex traffic conditions. Existing Graphical Convolutional Network (GCN)-based methods have demonstrated promising performance. However, most of them rely on graphs with predefined topologies, which leads to the inability to fully exploit the dynamic spatio-temporal correlation of traffic data and to explain the motivation behind traffic flows. Moreover, computing high-order Laplacian matrices in GCN-based methods is time-consuming and prone to be over-smoothing. In this paper, we propose a novel Semantics-aware Dynamic Graph Convolutional Network (SDGCN) for traffic flow forecasting to address the above problems. In particular, a sparse hidden Markov model based on state sharing is introduced to extract patterns of traffic flows from the sparse trajectory data, and learn the latent states of roads and transitions between them. As a result, we can adaptively construct dynamic road network Laplacian matrices by jointly considering the motivation of users. GCN is then applied to capture spatial features, while Gated Recurrent Unit (GRU) is employed to capture temporal features. We evaluated the system performance on two real traffic datasets. Experimental results show that the proposed method not only outperforms existing traffic flow forecasting methods in terms of prediction accuracy, but also obtains acceptable time complexity and better interpretation of the generated Laplace matrices, which can be used for traffic flow prediction in large cities and insight into the causes of traffic flow and traffic congestion.



The code consists of two parts. The first part is dynamic graph generation module, and the second part is graph convolutional network and temporal gate recursion predictor.In the first step, the moving graph module is used to extract the features of the trajectory data and generate the dynamic graphs of each time slice, and then the generated dynamic graphs are input into the second part to learn the spatio-temporal features, and finally output the prediction results. Moreover, the PyTorch version of dynamic graph generation，graph convolutional network and temporal gate recursion predictor are on the way~

##For the  dynamic graph generation module,

## Requirements:
* Eclipse IDE for Java Developers (includes Incubating components)Version: 2020-12 (4.18.0)
* Java(TM) SE Runtime Environment (build 1.8.0_291-b10)

## Run the demo:
SDGCN\State-sharing Sparse HMM\code\src\demo.java


##For the second part,

## Requirements:
* tensorflow
* scipy
* numpy
* matplotlib
* pandas
* math
* seaborn
* sklearn

## Run the demo:
SDGCN\SDGCN-master\SDGCN\main.py


***About dataset:

***Attention:
Data source: Didi Chuxing GAIA Initiative, because the DiDi company's data pact, we cannot publish the original or other data. If you want to obtain the data set, please apply for the authorization of didi company https://gaia.didichuxing.com, and inform us by email, we will send the dataset to you.


Please leave a message in the issue area with English or Chinese. You can also email me!
