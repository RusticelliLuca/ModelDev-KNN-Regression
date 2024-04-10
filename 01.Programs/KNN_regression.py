import pandas as pd 
import numpy as np 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV

############ Prepocessing ############ 

#paths
file_in =  "portfolio.csv"
path_in = "https://raw.githubusercontent.com/RusticelliLuca/ModelDev-KNN-Regression/main/Data/{}".format(file_in)

#import portfolio

data = pd.read_csv(path_in)

############ Definition of functions ############ 

#define function to compute KNN prediction

def KNN(K,y,data, regressors):
    """Params:
    - K : the K-number (of neighbours) to utilize for the estimation of the target variable
    - y: name of the column of the target,
    - data: dataframe-like object to manipulate during boostrapping and to which apply the function,
    - regressors: list of column names for the regressors of the algorithm.
    """
    #trasformation in specific arrays
    X_train = data[regressors].values
    y_train = data[y].values
    # fake array to predict the value of the target
    X_test = np.array([1,1,1,1,1,1]).reshape((1, -1))
    # Make predictions
    predictions = KNeighborsRegressor(n_neighbors=K).fit(X_train, y_train).predict(X_test)
    return predictions

#selecton of the number of neighbours K

def K_selection(y,list_k,data, regressors):
    """Params:
    - y: name of the column of the target,
    - list_k : the list from which select the K-number (of neighbours) to utilize for the estimation,
    - data: dataframe-like object to manipulate during boostrapping and to which apply the function,
    - regressors: list of column names for the regressors of the algorithm.
    """
    #define the training data
    X_train = data[regressors].values
    y_train = data[y].values
    #call the algorithm class
    knn = KNeighborsRegressor()
    param_grid = {'n_neighbors': list_k}
    # Perform grid search
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='neg_mean_absolute_error')
    # if wanted also 'neg_mean_squared_error')
    #cv=5 by default of the CV procedure
    grid_search.fit(X_train, y_train)
    # Get the best parameter
    best_k = grid_search.best_params_['n_neighbors']
    return best_k

#function to run the boostrapping procedure for N simulation times

def boot_se_estimation(list_k,y,n_bootr, data, boostr_size=1, shuffle=True, seed=0):
    """Params:
    - list_k : possible value for the seelction of the K-number (of naighbours)
    - n_boostr: number of boostrapped sample to simulate,
    - data: array-like object to manipulate during boostrapping and to which apply the function,
    - boostr_size: % of the obs of the input df which are extracted in the boostrapped sub-sample
    - shuffle: boolean value to indicate if consider replacement or not in the extraction of the
        boostrapped sample.
    - seed : number for the random number generator
    """
    #seed
    rng = np.random.default_rng(seed)
    #definition of the K neighbours number
    k = K_selection(y,list_k,data, regressors)
    #start of the boostrapping procedure
    #identification of the number of obs to extract
    n_obs = round(boostr_size*data.shape[0]) #n of rows 
    # for loop
    array = np.empty([n_bootr,1]) #initialize an empty array
    for n in range(0,n_bootr):
        idx = rng.choice(data.shape[0],n_obs, replace=shuffle) 
            #first param is an array like or a number from which extract 
            #second param is the number of obs to extract
            #replace -> shuffle or not
        data_resampled = data.loc[idx,:]
        array[n]=KNN(k,y,data_resampled, regressors)[0]
    #computation of the SE
    return np.mean(array, axis=0), np.std(array, axis=0), k

#column names of inputs
regressors=[ 
  "flag_1_2",
  "flag_1_3",
  "flag_1_4",
  "flag_2_3",
  "flag_2_4",
  "flag_3_4"
  ]

############ definition of paramters ############ 

#list of possible number of neighbours to consider in the estimation
list_k = [10,25,50,100,250,500,1000] 
drivers_list =  ["DTI","DSTI","LTI","LSTI"]

#to not overwrite the initial dataset and not have NaN values
data_copy = data.copy().dropna(subset=drivers_list).reset_index() 

#multiplicator factor to consider the std of the final result
multiplic_factor = 1 

#number of simulation of each risk threshold estimation
n_of_simulations = 100


############  initialization of the estimation for every LOM ############ 

#initial empty df to store results
df_final = pd.DataFrame(columns=["DRIVER","FINAL_THR","MEAN","STD","K"])

#for loop for the risk drivers
for driver in drivers_list:
    dict_={}
    mean, std, k =boot_se_estimation(list_k,driver,n_of_simulations, \
                                  data_copy, boostr_size=1, shuffle=False) 
    dict_["DRIVER"]=driver
    dict_["FINAL_THR"] = mean + multiplic_factor * std 
    dict_["MEAN"] = mean
    dict_["STD"] = std
    dict_["K"] = k
    df_final=pd.concat([df_final, pd.DataFrame(dict_)], ignore_index=True)

#show results

print(df_final)
