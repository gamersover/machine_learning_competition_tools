import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import catboost
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, mean_squared_error

# TODO: add sklearn model, evaluate accuracy

def train_lgb_cv(train_x, train_y, params, test_x=None, n_flods=5, return_fscore=False, features_name=None):
	"""
	params: {
	'learning_rate':0.02, 
	'boosting_type':'gbdt', 
	'objective':'binary', 
	'metric':'auc',
	'max_depth':7,  #5
	'num_leaves':120, #32
	'feature_fraction':0.7, 
	'min_data_in_leaf':400, #250
	'bagging_fraction':0.7, 
	'lambda_l2':100,
	'lambda_l1':0,
	'bagging_freq':5,
	'seed':0,
	"is_unbalance":True,
	"verbose":-1
	} 
	"""
	if return_fscore:
		featurs_name = train_x.columns if features_name is None else features_name
		fscore = pd.DataFrame()

	kf = KFold(n_splits=n_flods)
	lgb_flod = np.zeros((train_x.shape[0]))
	
	if test_x is not None:
		lgb_pred = np.zeros((test_x.shape[0], n_flods))
	
	for i, (trn_idx, val_idx) in enumerate(kf.split(train_x, train_y)):
		print("flod: [{}]".format(i))
		
		x_trn, y_trn = train_x[trn_idx], train_y[trn_idx]
		x_val, y_val = train_x[val_idx], train_y[val_idx]
		trn_dataset = lgb.Dataset(x_trn, y_trn)
		val_dataset = lgb.Dataset(x_val, y_val)
		
		model = lgb.train(params, trn_dataset, num_boost_round=20000, valid_sets=(trn_dataset, val_dataset),
			            early_stopping_rounds=50, verbose_eval=50)
		lgb_flod[val_idx] = model.predict(x_val, num_iteration=model.best_iteration)

		if test_x is not None:
			lgb_pred[:, i] = model.predict(test_x, num_iteration=model.best_iteration)

		if return_fscore:
			score = pd.DataFrame({"feature": features_name, "score": model.feature_importance()})
			fscore = pd.concat([fscore, score], ignore_index=True)

	return lgb_flod, lgb_pred if test_x is not None else None, fscore if return_fscore else None


def train_xgb_cv(train_x, train_y, params, test_x=None, n_flods=5, return_fscore=False):
	"""
	params: {
    'booster': 'gbtree', #  'dart' # 'rank:pairwise'对排序友好
    'objective': 'binary:logistic', # 'objective': 'multi:softmax', 'num_class': 3,
    'eta': 0.1,
    'max_depth': 7,

    'gamma': 0,
    'min_child_weight': 1,

    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'alpha': 0,
    'lambda': 1,

    'scale_pos_weight': 1,
    'eval_metric': 'auc', # 'rmse'
    
    'silent' : 1,
    'nthread': 16,
    'seed': 888,
	}
	"""
	kf = KFold(n_splits=n_flods)
	xgb_flod = np.zeros((train_x.shape[0]))

	if return_fscore:
		fscore = pd.DataFrame()
	
	if test_x is not None:
		xgb_pred = np.zeros((test_x.shape[0], n_flods))
		test_x = xgb.DMatrix(test_x)
	
	for i, (trn_idx, val_idx) in enumerate(kf.split(train_x, train_y)):
		print("flod: [{}]".format(i))
		
		x_trn, y_trn = train_x[trn_idx], train_y[trn_idx]
		x_val, y_val = train_x[val_idx], train_y[val_idx]
		trn_dataset = xgb.DMatrix(x_trn, y_trn)
		val_dataset = xgb.DMatrix(x_val, y_val)
		
		model = xgb.train(params, trn_dataset, num_boost_round=20000, 
			            evals=[(trn_dataset, "train"), (val_dataset, "val")],
			            early_stopping_rounds=50, verbose_eval=50)
		xgb_flod[val_idx] = model.predict(xgb.DMatrix(x_val), ntree_limit=model.best_iteration)

		if test_x is not None:
			xgb_pred[:, i] = model.predict(test_x, ntree_limit=model.best_iteration)

		if return_fscore:
			score = pd.Series(model.get_fscore()).reset_index()
			score.columns = ["feature", "score"]
			fscore = pd.concat([fscore, score], ignore_index=True)

	return xgb_flod, xgb_pred if test_x is not None else None, fscore if return_fscore else None


def train_catcls_cv(train_x, train_y, params, test_x=None, n_flods=5):
	"""
	params:{
	"depth": 7,
	"learning_rate": 0.1,
	"iterations": 10000,
	"eval_mertric":AUC",
	"l2_leaf_reg": 5,
	"rms": 0.5,
	"random_seed": 2019
	}
	"""
	model = catboost.CatBoostClassifier(**params)

	kf = KFold(n_splits=n_flods)
	cat_flod = np.zeros([train_x.shape[0]])

	if test_x is not None:
		cat_pred = np.zeros([test_x.shape[0], n_flods])

	for i, (trn_idx, val_idx) in enumerate(kf.split(train_x, train_y)):
	    print("flod: [{}]".format(i))
	    x_trn, y_trn = train_x[trn_idx], train_y[trn_idx]
	    x_val, y_val = train_x[val_idx], train_y[val_idx]
	    model.fit(x_trn, y_trn, eval_set=(x_val, y_val), use_best_model=True)
	    cat_flod[val_idx] = model.predict_proba(x_val)[:, 1]

	    if test_x is not None:
	    	cat_pred[:, i] = model.predict_proba(test_x)[:, 1]

	return cat_flod, cat_pred if test_x is not None else None


def train_catreg_cv(train_x, train_y, params, test_x=None, n_flods=5):
	"""
	params:{
	"depth": 7,
	"learning_rate": 0.1,
	"iterations": 10000,
	"eval_mertric":AUC",
	"l2_leaf_reg": 5,
	"rms": 0.5,
	"random_seed": 2019
	}
	"""
	model = catboost.CatBoostRegressor(**params)

	kf = KFold(n_splits=n_flods)
	cat_flod = np.zeros([train_x.shape[0]])

	if test_x is not None:
		cat_pred = np.zeros([test_x.shape[0], n_flods])

	for i, (trn_idx, val_idx) in enumerate(kf.split(train_x, train_y)):
	    print("flod: [{}]".format(i))
	    x_trn, y_trn = train_x[trn_idx], train_y[trn_idx]
	    x_val, y_val = train_x[val_idx], train_y[val_idx]
	    model.fit(x_trn, y_trn, eval_set=(x_val, y_val), use_best_model=True)
	    cat_flod[val_idx] = model.predict(x_val)

	    if test_x is not None:
	    	cat_pred[:, i] = model.predict(test_x)

	return cat_flod, cat_pred if test_x is not None else None


def train_skcls_cv(train_x, train_y, model, test_x=None, n_flods=5, metric=roc_auc_score):

	kf = KFold(n_splits=n_flods)
	model_flod = np.zeros([train_x.shape[0]])

	if test_x is not None:
		model_pred = np.zeros([test_x.shape[0], n_flods])

	for i, (trn_idx, val_idx) in enumerate(kf.split(train_x, train_y)):
	    print("flod: [{}]".format(i))
	    x_trn, y_trn = train_x[trn_idx], train_y[trn_idx]
	    x_val, y_val = train_x[val_idx], train_y[val_idx]
	    model.fit(x_trn, y_trn)
	    model_flod[val_idx] = model.predict_proba(x_val)[:, 1]
	    print("train-{0}\t{1:.5f}, eval-{0}\t{2:.5f}".format(metric.__name__, 
	    	metric(y_trn, model.predict_proba(x_trn)[:, 1]),
	    	metric(y_val, model.predict_proba(x_val)[:, 1])))

	    if test_x is not None:
	    	model_pred[:, i] = model.predict_proba(test_x)[:, 1]

	return model_flod, model_pred if test_x is not None else None


def train_skreg_cv(train_x, train_y, model, test_x=None, n_flods=5, metric=mean_squared_error):

	kf = KFold(n_splits=n_flods)
	model_flod = np.zeros([train_x.shape[0]])

	if test_x is not None:
		model_pred = np.zeros([test_x.shape[0], n_flods])

	for i, (trn_idx, val_idx) in enumerate(kf.split(train_x, train_y)):
	    print("flod: [{}]".format(i))
	    x_trn, y_trn = train_x[trn_idx], train_y[trn_idx]
	    x_val, y_val = train_x[val_idx], train_y[val_idx]
	    model.fit(x_trn, y_trn)
	    model_flod[val_idx] = model.predict(x_val)
	    print("train-{0}\t{1:.5f}, eval-{0}\t{2:.5f}".format(metric.__name__, 
	    	metric(y_trn, model.predict(x_trn)),
	    	metric(y_val, model.predict(x_val))))

	    if test_x is not None:
	    	model_pred[:, i] = model.predict(test_x)

	return model_flod, model_pred if test_x is not None else None