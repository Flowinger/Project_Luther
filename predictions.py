import pandas as pd
from dateutil.parser import parse
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor, AdaBoostRegressor, RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn import clone
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn import pipeline, manifold, preprocessing, feature_selection, decomposition

import logging
import xgboost as xgb
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt
import patsy
logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')

def split_df(dataframe, cols):
	'''
	Create dataframes for features and target variable
	INPUT: Dataframe and column names in list
	OUTPUT: Dataframe with features (X) and target variable dataframe (y) 
	'''

	X = df.loc[:,cols]
	y = df.loc[:, 'current_market_value']

	return X,y

def scale(x,y):
	'''
	INPUT: Dataframe with features (X) and target variable dataframe (y)  
	OUTPUT: Scaled dataframes
	'''

    X = preprocessing.robust_scale(x)
    y = preprocessing.robust_scale(y)

    return X,y

def run_linear_models(x,y):
    '''
    Get an overview of performances of different linear models.
    Linear models: Linear Regression, Ridge, 3x Lasso, ElasticNet
    INPUT: Dataframe with features (X) and target variable dataframe (y)
    OUTPUT: Scores and feature importances of each model
    '''
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    lr = LinearRegression()
    rd = Ridge()
    las2 = Lasso(alpha=0.00001)
    las1 = Lasso(alpha=.0001)
    las = Lasso(alpha=.001)
    en = ElasticNet()

    models = [lr,rd,las2,las1,las,en]

    for model in models:
        model.fit(X_train,y_train)
        sorted_features = sorted(zip(cols,model.coef_))
        
        print('Model: ' + str(model)[:25])
        print("Score: " + str(model.score(X_test,y_test)))    
        for feature in sorted_features:
            print(feature)
        print("")

def run_tree_models(x,y):
    '''
    Get an overview of performances of different tree models.
    Tree models: Decision tree, AdaBoost, Bagged tree
    INPUT: Dataframe with features (X) and target variable dataframe (y)
    OUTPUT: Scores of each tree model
    '''
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    dt = DecisionTreeRegressor()
    dt.fit(X_train, y_train)
    print('Decision Tree Score: ' + str(dt.score(X_test, y_test)))

    ada = AdaBoostRegressor(LinearRegression())
    ada.fit(X_train, y_train)
    print('AdaBoost Regressor Score: ' + str(ada.score(X_test, y_test)))

    # Train and Score Bagged Tree Regressor (ensemble learner)
    bagged_tree = BaggingRegressor(DecisionTreeRegressor())
    bagged_tree.fit(X_train, y_train)
    print('Bagged Tree Score: ' + str(bagged_tree.score(X_test, y_test)))

def random_forest(X, y,n_est):
    '''
    Run a random forest model.
	INPUT: Dataframe with features (X), target variable dataframe (y), number of estimators (parameter)
	OUTPUT: Score of random forest model
    '''
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    rfc = RandomForestRegressor(n_estimators=n_est)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X = (X - mean) / std
    clf = clone(rfc)
    clf = rfc.fit(X_train, y_train)

    scores = clf.score(X_test, y_test)

    return 'Random Forest Score: '+str(scores), dict(zip(cols2, clf.feature_importances_))

def extra_trees(X,y,n_est):
    '''
	INPUT: Dataframe with features (X), target variable dataframe (y), number of estimators (parameter)
	OUTPUT: Score of ExtaTrees model
    '''

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    ext = ExtraTreesRegressor(n_estimators=n_est)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X = (X - mean) / std
    clf = clone(ext)
    clf = ext.fit(X_train, y_train)

    scores = ext.score(X_test, y_test)

    return 'ExtraTrees Score: '+str(scores), dict(zip(cols2, clf.feature_importances_))

def poly_rf(x,y,degree):
	'''
	INPUT: Dataframe with features (X), target variable dataframe (y), polynomial degree (parameter)
	OUTPUT: Score of pipeline using Random forest with polynomial features.
    '''
    dgr = degree
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    est = make_pipeline(PolynomialFeatures(dgr),RandomForestRegressor(n_estimators=200,max_features='auto',min_samples_split=10))
    est.fit(X_train,y_train)
    print('Scores:',est.score(X_test,y_test))

def gradient_boosting(x,y,n_est):
	'''
	INPUT: Dataframe with features (X), target variable dataframe (y), polynomial degree (parameter)
	OUTPUT: Score of Gradient Boosting Regressor
    '''
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    gradient_boosting = GradientBoostingRegressor(n_estimators = n_est,max_features=3, 
                                         learning_rate = .15,
                                        max_depth = 2,min_samples_leaf=8, alpha=.5)
    gradient_boosting.fit(X_train,y_train)

    return gradient_boosting.score(X_test,y_test)

def xgb_regressor(x,y,n_est):
	'''
	INPUT: Dataframe with features (X), target variable dataframe (y), polynomial degree (parameter)
	OUTPUT: Score of XGB Regressor
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    model_xgb = xgb.XGBRegressor(n_estimators=n_est,learning_rate=0.15,max_depth=2)
	model_xgb.fit(X_train, y_train, eval_metric='rmse')
	
	return model_xgb.score(X_test,y_test)

def regression_pipeline(x,y,degree,pca_comps):
	'''
	INPUT: Dataframe with features (X), target variable dataframe (y), polynomial degree, PCA components
	OUTPUT: Score of regression pipeline using polynomial features and PCA
	'''
    # lets try using some feature tranforms, to our original X
    combined_features = pipeline.FeatureUnion([('poly',PolynomialFeatures(degree=degree)),
                                              ('pca', decomposition.PCA(n_components=pca_comps))])
    X_train,X_test,y_train,y_test = train_test_split(x,y,test_size = .3)
    
    # now lets lay out the steps of our model
    # First we do the feature transforms with our feature union defined above
    # Second we do feature selection with the built-in SelectFromModel
    # Third we train the actual model
    steps = [
    ('features', combined_features),
    ('feature_selection', feature_selection.SelectFromModel(Lasso(alpha=.5))),
    ('model', LinearRegression())]

    #but at this point we have only defined what reg, no training has happened yet
    regression_pipeline = pipeline.Pipeline(steps)
    # calling fit here calls fit on the entire pipeline which in turn executes all its members
    regression_pipeline.fit(X_train,y_train)
    # calling score/predict will call fit then transform on each object in the pipeline
    print(regression_pipeline.score(X_test,y_test))

def gridsearch_cv():
	'''
	Grid search with the regression pipeline using different polynomial degress and 
	numbers of PCA components.
	OUTPUT: Score and values of best parameters set
	'''
	
	parameters = {
	    'features__poly__degree': (2,3,4),
	    'features__pca__n_components': (2,3,4),
	}

	grid_search = GridSearchCV(regression_pipeline, parameters, n_jobs=-2, verbose=1)
	grid_search.fit(X_train,y_train)

	print("Best score: %0.3f" % grid_search.best_score_)
	print("Best parameters set:")
	best_parameters = grid_search.best_estimator_.get_params()
	for param_name in sorted(parameters.keys()):
	    print("\t%s: %r" % (param_name, best_parameters[param_name]))

def run(filepath_df, columns):
	# Run model
	df = pd.read_csv(filepath_df)
	cols = 
	X,y = split_df(df, cols)
	print(run_linear_models(X,y))

	# Run random forest model with different numbers of estimators as a parameter.
	for i in [10,30,50,100,200,300]:
	    print(random_forest(X,y,i))
	# Run ExtraTrees model with different numbers of estimators as a parameter.
	for i in [10,30,50,100,200,300]:
	    print(extra_trees(X,y,i))
	# Create a pipeline using Random forest with polynomial features.
	# Degree: 3
	poly_rf(X,y,3)
	# Run gradient boosting with different number of estimators.
	for i in [10,30,50,100,200,300]:
	    print(gradient_boosting(i))
	# Run XGBoosting with different number of estimators.
	for i in [10,30,50,100,200,300]:
	    print(xgb_regressor(i))
	# Run GridSearch CV
	print(gridsearch_cv())

features = ['age','total_transfer_proceeds','assists_season','cl_winner',\
'days_in_team','days_to_contract_expiry','international_exp','liga1','goals*mins']

run('Project_Luther_dataframe.csv', features)
