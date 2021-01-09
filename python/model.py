"""In this module, we ask you to define your pricing model, in Python."""


import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# TODO: import your modules here.
# Don't forget to add them to requirements.txt before submitting.


# DATA
#%%


df=pd.read_csv("training.csv")



df[["Min","Med1","Med2","Max"]]=pd.get_dummies(df.pol_coverage)
df[['Biannual', 'Monthly', 'Quarterly', 'Yearly']]=pd.get_dummies(df.pol_pay_freq)
df[['AllTrips', 'Professional', 'Retired', 'WorkPrivate']]=pd.get_dummies(df.pol_usage)
df[['Sex2_0', 'Sex2_F', 'Sex2_M']]=pd.get_dummies(df.drv_sex2)
df[['Diesel', 'Gasoline', 'Hybrid']]=pd.get_dummies(df.vh_fuel)
df[['Commercial', 'Tourism']]=pd.get_dummies(df.vh_type)

df["pol_payd"]=df["pol_payd"].replace("No",1).replace("Yes",0)
df["drv_drv2"]=df["drv_drv2"].replace("No",1).replace("Yes",0)
df["drv_sex1"]=df["drv_sex1"].replace("F",1).replace("M",0)
df["drv_age2"]=df["drv_age2"].fillna(0)
df["drv_age_lic2"]=df["drv_age_lic2"].fillna(0)

df=df.drop(["pol_coverage","pol_pay_freq","pol_usage",\
                "drv_sex2","vh_fuel","vh_type","vh_make_model"],axis=1)

df=df.dropna()
id_train, id_test= train_test_split(df["id_policy"].unique(),test_size=0.20, random_state=5)

df1=df[df["year"]==1].drop("year",axis=1).sort_values(by=["id_policy"]).reset_index(drop=True)
scaler1 = MinMaxScaler()
df1["claim_amount"] = pd.DataFrame(scaler1.fit_transform(np.array(df1["claim_amount"]).reshape(-1, 1)))

df2=df[df["year"]==2].drop("year",axis=1 ).sort_values(by=["id_policy"]).reset_index(drop=True)
scaler2 = MinMaxScaler()
df2["claim_amount"] = pd.DataFrame(scaler2.fit_transform(np.array(df2["claim_amount"]).reshape(-1, 1)))

df3=df[df["year"]==3].drop("year",axis=1).sort_values(by=["id_policy"]).reset_index(drop=True)
scaler3 = MinMaxScaler()
df3["claim_amount"] = pd.DataFrame(scaler3.fit_transform(np.array(df3["claim_amount"]).reshape(-1, 1)))

df4=df[df["year"]==4].drop("year",axis=1).sort_values(by=["id_policy"]).reset_index(drop=True)
scaler4 = MinMaxScaler()
df4["claim_amount"] = pd.DataFrame(scaler4.fit_transform(np.array(df4["claim_amount"]).reshape(-1, 1)))

claim_amount_train = np.array(df4[df4["id_policy"].isin(id_train)]["claim_amount"])
claim_amount_test = np.array(df4[df4["id_policy"].isin(id_test)]["claim_amount"])




label_con=['pol_no_claims_discount','pol_duration', 'pol_sit_duration',
        'drv_age1', 'drv_age_lic1','drv_age2','drv_age_lic2','vh_age',
        'vh_speed', 'vh_value', 'vh_weight','population','town_surface_area','claim_amount']

label_dis=["Min","Med1","Med2","Max",'Biannual', 'Monthly', 'Quarterly',\
        'Yearly','AllTrips', 'Professional', 'Retired', 'WorkPrivate',\
        'Sex2_0', 'Sex2_F', 'Sex2_M','Diesel', 'Gasoline', 'Hybrid','Commercial', 'Tourism']


#X_Training_Data=np.array(list(zip(df1.drop("claim_amount",axis=1).values.tolist(),
#                                  df2.drop("claim_amount",axis=1).values.tolist(),
#                                  df3.drop("claim_amount",axis=1).values.tolist())))


X_Training_Data=np.array(list(zip(df1[df1["id_policy"].isin(id_train)].drop("id_policy",axis=1).values.tolist(),
                                  df2[df2["id_policy"].isin(id_train)].drop("id_policy",axis=1).values.tolist(),
                                  df3[df3["id_policy"].isin(id_train)].drop("id_policy",axis=1).values.tolist())))


X_Test_Data=np.array(list(zip(df1[df1["id_policy"].isin(id_test)].drop("id_policy",axis=1).values.tolist(),
                              df2[df2["id_policy"].isin(id_test)].drop("id_policy",axis=1).values.tolist(),
                              df3[df3["id_policy"].isin(id_test)].drop("id_policy",axis=1).values.tolist())))

      


# model   
#%%    
rnn = Sequential()

#Adding our first LSTM layer

rnn.add(LSTM(units = 1, input_shape = (3, X_Training_Data.shape[2])))

#Perform some dropout regularization

rnn.add(Dropout(0.2))

#Adding our output layer

rnn.add(Dense(units = 1))

    
#Compiling the recurrent neural network

rnn.compile(optimizer = 'adam', loss = 'mean_squared_error')

#Training the recurrent neural network

rnn.fit(X_Training_Data, claim_amount_train, epochs = 100)

predictions = rnn.predict(X_Test_Data)

plt.plot(predictions)

unscaled_predictions = scaler4.inverse_transform(predictions)

unscaled_resutados = np.array(scaler4.inverse_transform(claim_amount_test.reshape(-1, 1)))

# Feel free to create any number of other functions, constants and classes to use
# in your model (e.g., a preprocessing function).



def fit_model(X_raw, y_raw):
    """Model training function: given training data (X_raw, y_raw), train this pricing model.

    Parameters
    ----------
    X_raw : Pandas dataframe, with the columns described in the data dictionary.
        Each row is a different contract. This data has not been processed.
    y_raw : a Numpy array, with the value of the claims, in the same order as contracts in X_raw.
        A one dimensional array, with values either 0 (most entries) or >0.

    Returns
    -------
    self: this instance of the fitted model. This can be anything, as long as it is compatible
        with your prediction methods.

    """

    # TODO: train your model here.
    

    
    return np.mean(y_raw)  # By default, training a model that returns a mean value (a mean model).



def predict_expected_claim(model, X_raw):
    """Model prediction function: predicts the expected claim based on the pricing model.

    This functions estimates the expected claim made by a contract (typically, as the product
    of the probability of having a claim multiplied by the expected cost of a claim if it occurs),
    for each contract in the dataset X_raw.

    This is the function used in the RMSE leaderboard, and hence the output should be as close
    as possible to the expected cost of a contract.

    Parameters
    ----------
    model: a Python object that describes your model. This can be anything, as long
        as it is consistent with what `fit` outpurs.
    X_raw : Pandas dataframe, with the columns described in the data dictionary.
        Each row is a different contract. This data has not been processed.

    Returns
    -------
    avg_claims: a one-dimensional Numpy array of the same length as X_raw, with one
        expected claim per contract (in same order). These expected claims must be POSITIVE (>0).
    """

    # TODO: estimate the expected claim of every contract.

    return np.full( (len(X_raw.index),), model )  # Estimate that each contract will cost 114 (this is the naive mean model). You should change this!



def predict_premium(model, X_raw):
	"""Model prediction function: predicts premiums based on the pricing model.

	This function outputs the prices that will be offered to the contracts in X_raw.
	premium will typically depend on the expected claim predicted in 
	predict_expected_claim, and will add some pricing strategy on top.

	This is the function used in the expected profit leaderboard. Prices output here will
	be used in competition with other models, so feel free to use a pricing strategy.

	Parameters
	----------
	model: a Python object that describes your model. This can be anything, as long
	    as it is consistent with what `fit` outpurs.
	X_raw : Pandas dataframe, with the columns described in the data dictionary.
		Each row is a different contract. This data has not been processed.

	Returns
	-------
	prices: a one-dimensional Numpy array of the same length as X_raw, with one
	    price per contract (in same order). These prices must be POSITIVE (>0).
	"""

	# TODO: return a price for everyone.

	return predict_expected_claim(model, X_raw)  # Default: price at the pure premium with no pricing strategy.



def save_model(model):
	"""Saves this trained model to a file.

	This is used to save the model after training, so that it can be used for prediction later.

	Do not touch this unless necessary (if you need specific features). If you do, do not
	 forget to update the load_model method to be compatible.

	Parameters
	----------
	model: a Python object that describes your model. This can be anything, as long
	    as it is consistent with what `fit` outpurs."""

	with open('trained_model.pickle', 'wb') as target:
		pickle.dump(model, target)




def load_model():
	"""Load a saved trained model from the file.

	   This is called by the server to evaluate your submission on hidden data.
	   Only modify this *if* you modified save_model."""

	with open('trained_model.pickle', 'rb') as target:
		trained_model = pickle.load(target)
	return trained_model
