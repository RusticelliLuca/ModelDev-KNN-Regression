# ModelDev-KNN-Regression

This real-world project demonstrates the use of KNN (K-Nearest Neighbors) regression to estimate a threshold value of four different LOM indicators, based on a given development dataset, in order to indentify debtors more vulnerable as the ones with one or more LOM values higher than the defined thresholds. 

The Loan Origination Metrics (LOM) serve as pivotal indicators of a debtor's creditworthiness when seeking a loan from a financial institution, determined by assessing their income level and expenses. An example is the Debt to Income (DTI) ratio, calculated as the monthly debt payments (including both new and existing obligations) divided by an estimate of the monthly net income.

KNN regression is a type of supervised learning algorithm used for regression tasks, where the output is a continuous value. In this project, the KNN regression is utilized to predict a threshold value of the continuous LOM based on some flag variables created on credit risk charachteristcs as rating downgrade, staging IFRS9, forborne, and presence in watchlist.

The development dataset comprises a fictional sample, wherein each row represents an observation of a debtor who applied for a loan in previous years and for whom at least one year has elapsed—a prerequisite for assessing their credit risk metrics, such as downgrade, one year after origination. More information are presented in the README.txt inside the "Data" folder.

## Getting Started

To run this project gloabally it is possible going to the "Actions" panel and execute manually the workflow "run KNN_regression.py".

To run this project locally, follow these steps:
  1. Clone this repository to your local machine.
  2. Install the required dependencies listed in the `requirements.txt` file.
  3. Modify the path_in variable in `main.py` script based on the local path where the csv dataset contained in the folder "Data" is saved. 
  4. Run the `main.py` script to execute the KNN regression algorithm and identify the threshold for the dataset.

