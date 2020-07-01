##############################################################
#
# 500 Days of AAPL data used to predict 50 days in the future
#
#-------------------------------------------------------------
# Author : Dhruv Oberoi
# Dash App for Stock Price Prediction using
# Linear Regression, Ridge Regression, Lasso, SVR (RBF Kernel),
# Random Forest and XGBoost
#-------------------------------------------------------------
# Next Iterations:
#
# Fix date issue ; Dates are pushed forward 50 days
# Reduce overuse of df's
# Introduce range slider for date to alter training dataset
# Print and alter based on results from GridSearchCV
#
##############################################################
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, Event

import plotly
import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools

import matplotlib.pyplot as plt
import datetime
import csv
import quandl, math
import numpy as np
import pandas as pd
import datetime

from sklearn import preprocessing, cross_validation, svm
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ARDRegression
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import sparse_random_matrix
from matplotlib import style
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

#import plotly
#plotly.tools.set_credentials_file(username='dhruv.oberoi', api_key='API KEY')

quandl.ApiConfig.api_key = 'PASTE YOUR API KEY HERE'
df = quandl.get("EOD/AAPL")
###########################################################################

dataset = df[['Adj_Open','Adj_High',  'Adj_Low',  'Adj_Close', 'Adj_Volume']].copy()
dataset = dataset.iloc[pd.np.r_[:,-501:-1]]

dataset['HL_PCT'] = (dataset['Adj_High'] - dataset['Adj_Low']) / dataset['Adj_Low'] * 100.0
dataset['PCT_change'] = (dataset['Adj_Close'] - dataset['Adj_Open']) / dataset['Adj_Open'] * 100.0

dataset = dataset[['Adj_Close', 'HL_PCT', 'PCT_change', 'Adj_Volume']]

pred_feature = 'Adj_Close'
dataset.fillna(value=99999, inplace=True)

no_of_var = int(math.ceil(0.1 * len(dataset)))

dataset['label'] = dataset[pred_feature].shift(-no_of_var)

x = np.array(dataset.drop(['label'], 1))
x = preprocessing.scale(x)
x_small = x[-no_of_var:]
x_small = x[-no_of_var:]
x = x[:-no_of_var]

dataset.dropna(inplace=True)
y = np.array(dataset['label'])

x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.2)

dataset_2 = df[['Adj_Open','Adj_High',  'Adj_Low',  'Adj_Close', 'Adj_Volume']].copy()
dataset_2 = dataset_2.iloc[pd.np.r_[:,-51:-1]]
df = df.iloc[pd.np.r_[:,-801:-1]]


###########################################################################

last_date = df.iloc[-1].name
#name of last date
last_unix = last_date.timestamp()
one_day = 86400
#number of seconds in a day
next_unix = last_unix + one_day
#next day...
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################

model1 = svm.LinearSVR()
model1.fit(x_train, y_train)
confidence1 = model1.score(x_test, y_test)
predict_1 = model1.predict(x_small)
dataset['Predict_Linear'] = np.nan
print('Score for Linear Reg: :',confidence1)
print('\n')

for i in predict_1:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    dataset.loc[next_date] = [np.nan for _ in range(len(dataset.columns)-1)]+[i]
####################################################################################

model2 = svm.SVR(kernel = 'rbf', C= 100, gamma= 0.06)
model2.fit(x_train, y_train)
confidence2 = model2.score(x_test, y_test)
predict_2 = model2.predict(x_small)
dataset['Predict_RBF'] = np.nan
print('Score for RBF Reg: :',confidence2)
print('\n')


for i in predict_2:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    dataset.loc[next_date] = [np.nan for _ in range(len(dataset.columns)-1)]+[i]

####################################################################################

model5 = RandomForestRegressor(n_estimators = 150, random_state = 0)
model5.fit(x_train, y_train)
confidence5 = model5.score(x_test, y_test)
predict_5 = model5.predict(x_small)
dataset['Predict_RF'] = np.nan
print('Score for RF Reg: :',confidence5)
print('\n')

for i in (predict_5):
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    dataset.loc[next_date] = [np.nan for _ in range(len(dataset.columns)-1)]+[i]

####################################################################################

model6 = XGBRegressor(max_depth=1, learning_rate=0.05, n_estimators=200,
                      objective="reg:linear", booster='gbtree', n_jobs=-1,
                      nthread=None, gamma=0, min_child_weight=1, max_delta_step=0,
                      subsample=1, colsample_bytree=1, random_state=0)
model6.fit(x_train, y_train)
confidence6 = model6.score(x_test, y_test)
predict_6 = model6.predict(x_small)
dataset['Predict_XGBR'] = np.nan
print('Score for XGBR Reg: :',confidence6)
print('\n')

for i in (predict_6):
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    dataset.loc[next_date] = [np.nan for _ in range(len(dataset.columns)-1)]+[i]

####################################################################################

model7 = Ridge(alpha=1, fit_intercept = True, tol = 0.001, random_state = 0, solver = 'saga')
model7.fit(x_train, y_train)
confidence7 = model7.score(x_test, y_test)
predict_7 = model7.predict(x_small)
dataset['Predict_Ridge'] = np.nan
print('Score for Ridge Reg: :',confidence7)
print('\n')

for i in (predict_7):
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    dataset.loc[next_date] = [np.nan for _ in range(len(dataset.columns)-1)]+[i]

####################################################################################

model8 = Lasso(alpha=0.1,fit_intercept = True, tol = 0.01, random_state = 0, selection = 'cyclic')
model8.fit(x_train, y_train)
confidence8 = model8.score(x_test, y_test)
predict_8 = model8.predict(x_small)
dataset['Predict_Lasso'] = np.nan
print('Score for Lasso Reg: :',confidence8)
print('\n')

for i in (predict_8):
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    dataset.loc[next_date] = [np.nan for _ in range(len(dataset.columns)-1)]+[i]

###########################################################################
###########################################################################
###########################################################################
app = dash.Dash(__name__)

layout = go.Layout(
yaxis=dict(
    domain=[1, 1]
),
legend=dict(
    traceorder='reversed'
),
yaxis2=dict(
    domain=[1, 1]
),
yaxis3=dict(
    domain=[1, 1]
)
)
fig = go.Figure(layout = layout)

###########################################################################
###########################################################################
###########################################################################
all_options = {
    'Linear Regression': ['lin'],
    'Ridge Regression': ['rid'],
    'Lasso Regression': ['las'],
    'Random Forest Regression': ['rf'],
    'XGBOOST Regression': ['xgb'],
    'SVR Regression': ['rbf']
            }

app.layout = html.Div([
        html.Hr(),
    	dcc.Dropdown(
            id='models-dropdown', 
            options=[{'label': k, 'value': k} for k in all_options.keys()],
            value = 'Linear Regression',
            multi = False),
        html.Hr(),
    	html.Div(id='show-live'),
        html.Div([dcc.Graph(id='plot', figure = fig)]),
		dcc.Interval(id='interval-component', interval=1*10000) # in milliseconds
					])
###########################################################################
###########################################################################

@app.callback(
    dash.dependencies.Output(component_id = 'plot', component_property='figure'),
    [dash.dependencies.Input(component_id = 'models-dropdown', component_property = 'value')],
    events=[Event('interval-component', 'interval')]
    		)

def update_val(value):

    trace1 = go.Scatter(x=df.index,y=df.Adj_Close)
#    trace2 = go.Scatter(x=dataset.index,y=model_sel2)
#    fig = go.Figure(data=[trace1,trace2],layout = layout)

    if value == 'Ridge Regression':
        print('{} is working'.format(value))
        print("\n")
        model_sel2 = dataset.Predict_Ridge
        trace2 = go.Scatter(x=dataset.index,y=model_sel2)
        fig = go.Figure(data=[trace1,trace2],layout = layout)
        print(model_sel2.tail(2))

    elif value == 'Lasso Regression':
        print('{} is working'.format(value))
        print("\n")
        model_sel2 = dataset.Predict_Lasso
        trace2 = go.Scatter(x=dataset.index,y=model_sel2)
        fig = go.Figure(data=[trace1,trace2],layout = layout)
        print(model_sel2.tail(2))

    elif value == 'XGBOOST Regression':
        print('{} is working'.format(value))
        print("\n")
        model_sel2 = dataset.Predict_XGBR
        trace2 = go.Scatter(x=dataset.index,y=model_sel2)
        fig = go.Figure(data=[trace1,trace2],layout = layout)
        print(model_sel2.tail(2))

    elif value == 'Random Forest Regression':
        print('{} is working'.format(value))
        print("\n")
        model_sel2 = dataset.Predict_RF
        trace2 = go.Scatter(x=dataset.index,y=model_sel2)
        fig = go.Figure(data=[trace1,trace2],layout = layout)
        print(model_sel2.tail(2))

    elif value == 'SVR Regression':
        print('{} is working'.format(value))
        print("\n")
        model_sel2 = dataset.Predict_RBF
        trace2 = go.Scatter(x=dataset.index,y=model_sel2)
        fig = go.Figure(data=[trace1,trace2],layout = layout)
        print(model_sel2.tail(2))

    else:
        print('{} is working'.format(value))
        print("\n")
        model_sel2 = dataset.Predict_Linear
        trace2 = go.Scatter(x=dataset.index,y=model_sel2)
        fig = go.Figure(data=[trace1,trace2],layout = layout)
        print(model_sel2.tail(1))
	
    return fig

###########################################################################

@app.callback(dash.dependencies.Output(component_id = 'show-live', component_property='children'),
    [dash.dependencies.Input(component_id = 'models-dropdown', component_property = 'value')],)

def set_display_children(modelsdropdown):
    if modelsdropdown is None:
        return ''
    else:
        return '{} has been selected-ish'.format(modelsdropdown)

###########################################################################

if __name__ == "__main__":
    app.run_server(debug=True)

###########################################################################
###########################################################################
