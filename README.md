[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/U9FTc9i_)

# Machine Learning Project 1 :  MICHD Classification

This is part of the Project 1 of the CS433-Machine Learning course given at the EPFL Fall 2023.

## Team KER : 
- Kaan Uçar
- Elias Nicolas Naha
- Riccardo Carpineto

## Project Description 

The aim of the project is to predict MICH risk based on personal features. To perform this classification task we need to perfom the following :
- preprocessing health related data
- implement basic regression models 
- hyperparameter tuning

We then select the most performing one. The performance is measured by the F1 score which is a good perfomance value for imbalanced data.
In our best run we achieved 0.419 F1 score and 0.869 accuracy using Ridge regression.

## Structure of Repository :

- `implementations.py` : contains implementations of Linear regression using gradient descent,Linear regression using stochastic gradient descent, Least squares regression using normal equations, Ridge regression using normal equations, Logistic regression using gradient descent, Regularized logistic regression using gradient descent. This file also contains all the helper functions (load, preprocessing, submission...)

- `run.ipynb` : executable file which reproduces our best score on AICrowd

- `data` : folder containing all the data

- `code` : folder containing all the runs for the different methods implemented.

## Instructions to run :

To reproduce our best score it is only needed to run the file `run.ipynb` with the presence of the `implementation.py` file and `data` folder. The prediction will be saved in a file named `ridge_reg.csv` under the folder `data`.
