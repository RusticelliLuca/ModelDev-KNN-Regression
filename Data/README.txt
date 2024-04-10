The file CSV contains an example dataset to run the KNN regression.
The data are fictitious but coherent with the aim of the algorithm.

In particular:

- the first 6 columns (i.e. flag_1_2) are related to credit risk indicators as rating downgrade, stage IFRS9, forborne, watchlist
There is a flag for each pair of risk drivers where it is equal to 1 when both risk factor are present (i.e. both forborne and watchlist) and 0 otherwise.

- the last 4 columns present the value of our LOM indicator, defined based on the information provided at the origination of the contract with the bank.
