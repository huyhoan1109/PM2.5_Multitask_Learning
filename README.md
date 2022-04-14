# PM2.5's concentration Multitask-Learning
## Purpose
PM2.5 is an air pollutant that has great concern for people's health when levels in air are high. This kind of pollutant usually has high concentration in big city like Beijing, London, ...

Base on the multitask-learning neural network created by the Kyoto University's professor, Yoneda Minoru, I developed the project with a scheme to be able to predict PM2.5's concentration among multiple sites in urban areas. 

## Abstract
The dataset is available on this [link](https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data) 

Firstly, the model extracts the necessary informations of other factors like O2's concentration, amount of rain, temperature, wind speed, ... After learning the above latent data, it is cooperated with PM2.5's time series to predict future PM2.5's concentration.

## Tech
The project requires libraries and frameworks to be installed to work properly:
- [Pytorch](https://pytorch.org)
- [Sklearn](https://scikit-learn.org/stable/)

## Future
- [Deploying on a server]
- [Cooperating with other kinds of models]
