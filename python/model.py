"""In this module, we ask you to define your pricing model, in Python."""

import pickle
import numpy as np
import pandas as pd

# TODO: import your modules here.
# Don't forget to add them to requirements.txt before submitting.



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
