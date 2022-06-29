import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame
# metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error
# cross validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

df = pd.read_csv('outputs_ML_hires.csv',sep=';')  # file created by RTM TUV
#df = pd.read_csv('/content/drive/MyDrive/Pesquisa/SolUVCC/outputs_TUV.csv',sep=';')

df.head(2)
indep = df.iloc[:, 0:3].values
Ery = df.iloc[:,5].values

# Preparing training and testing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
x_treino, x_teste, y_treino, y_teste = train_test_split(indep, Ery, test_size = 0.3, random_state = 0)

########################################
# MULTIPLE LINEAR REGRESSION
########################################
multipla = LinearRegression()
multipla.fit(x_treino, y_treino)
print("Equation: {:.5f} + ({:.5f})*RM + ({:.5f})*LSTAT + ({:.5f})*PTRATIO".format(multipla.intercept_, multipla.coef_[0], multipla.coef_[1], multipla.coef_[2]))
multipla.score(x_treino, y_treino)      # Determination coefficient

### TEST
previsoes = multipla.predict(x_teste)
multipla.score(x_teste, y_teste)        # Determination coefficient

### METRICS
MAE_RLM = abs(y_teste - previsoes).mean()       # Absolute error
meanMAE_RLM = mean_absolute_error(y_teste, previsoes)       # Mean absolute error
MSE_RLM = mean_squared_error(y_teste, previsoes)        # Mean squared error
RMSE_RLM = np.sqrt(mean_squared_error(y_teste, previsoes))      # Root of the mean squared error

### CROSS VALIDATION
# Separating data into folds
kfold = KFold(n_splits = 50, shuffle=True, random_state = 5)

# Generating the model
from sklearn.linear_model import LinearRegression
modelo = LinearRegression()
resultado = cross_val_score(modelo, indep, Ery, cv = kfold)
resultado_RLM = resultado

# We use the mean and standard deviation
print("Average Coefficient of Determination: %.2f%%" % (resultado.mean() * 100.0))

########################################
# REGRESSION BY SUPPORT VECTORS (SVM)
########################################
from sklearn.svm import SVR
SVR = SVR(kernel='rbf')
SVR.fit(x_treino,y_treino)

# Padronizando as escalas
from sklearn.preprocessing import StandardScaler
x_scaler = StandardScaler()
x_treino_scaler = x_scaler.fit_transform(x_treino)
x_teste_scaler = x_scaler.transform(x_teste)
y_scaler = StandardScaler()
y_treino_scaler = y_scaler.fit_transform(y_treino.reshape(-1,1))
y_teste_scaler = y_scaler.transform(y_teste.reshape(-1,1))

from sklearn.svm import SVR
SVR2 = SVR(kernel='rbf')
SVR2.fit(x_treino_scaler, y_treino_scaler.ravel())
SVR2.score(x_treino_scaler, y_treino_scaler)        # Determination coefficient

### TEST
SVR2.score(x_teste_scaler, y_teste_scaler)
previsoes_teste = SVR2.predict(x_teste_scaler)

### Reversing the transformation
y_teste_inverse = y_scaler.inverse_transform(y_teste_scaler)
previsoes_inverse = y_scaler.inverse_transform(previsoes_teste.reshape(-1, 1))

### METRICS
MAE_SVM = abs(y_teste - previsoes_teste).mean()
meanMAE_SVM = mean_absolute_error(y_teste, previsoes_teste)
MSE_SVM = mean_squared_error(y_teste, previsoes_teste)
RMSE_SVM = np.sqrt(mean_squared_error(y_teste, previsoes_teste))

### Cross validation
from sklearn.preprocessing import StandardScaler
x = StandardScaler()
independente_scaler = x.fit_transform(indep)
y = StandardScaler()
dependente_scaler = y.fit_transform(Ery.reshape(-1,1))

# Generating the model
from sklearn.svm import SVR
modelo = SVR(kernel='rbf')
resultado = cross_val_score(modelo, independente_scaler, dependente_scaler.ravel(), cv = kfold)
resultado_SVR = resultado

print("Average Coefficient of Determination: %.2f%%" % (resultado.mean() * 100.0))

########################################
# REGRESSION BY DECISION TREES
########################################
from sklearn.tree import DecisionTreeRegressor
arvore = DecisionTreeRegressor(max_depth=5, random_state=10)
arvore.fit(x_treino, y_treino)
arvore.score(x_treino, y_treino)        # Determination coefficient

### TEST
arvore.score(x_teste, y_teste)
previsoes_teste = arvore.predict(x_teste)

### METRICS
MAE_AD = abs(y_teste - previsoes_teste).mean()
meanMAE_AD = mean_absolute_error(y_teste, previsoes_teste)
MSE_AD = mean_squared_error(y_teste, previsoes_teste)
RMSE_AD = np.sqrt(mean_squared_error(y_teste, previsoes_teste))

### CROSS VALIDATION
from sklearn.tree import DecisionTreeRegressor
modelo = DecisionTreeRegressor(max_depth=5, random_state=10)
resultado = cross_val_score(modelo, indep, Ery, cv = kfold)
resultado_AD = resultado

print("Average Coefficient of Determination: %.2f%%" % (resultado.mean() * 100.0))

########################################
# REGRESSION WITH RANDOM FOREST
########################################
from sklearn.ensemble import RandomForestRegressor
random = RandomForestRegressor(n_estimators=60, criterion='squared_error', max_depth=5, random_state = 10)
random.fit(x_treino, y_treino)
random.score(x_treino, y_treino)        # Determination coefficient

### TESTE
random.score(x_teste, y_teste)
previsoes_teste = random.predict(x_teste)

### METRICS
MAE_RF = abs(y_teste - previsoes_teste).mean()
meanMAE_RF = mean_absolute_error(y_teste, previsoes_teste)
MSE_RF = mean_squared_error(y_teste, previsoes_teste)
RMSE_RF = np.sqrt(mean_squared_error(y_teste, previsoes_teste))

### CROSS VALIDATION
from sklearn.ensemble import RandomForestRegressor
modelo = RandomForestRegressor(n_estimators=60, criterion='squared_error', max_depth=5, random_state = 10)
resultado = cross_val_score(modelo, indep, Ery, cv = kfold)
resultado_RF = resultado

print("Average Coefficient of Determination: %.2f%%" % (resultado.mean() * 100.0))

########################################
# REGRESSION WITH XGBOOST
########################################
from xgboost import XGBRegressor
xgboost = XGBRegressor(n_estimators=180, max_depth=3, learning_rate=0.05, objective="reg:squarederror", random_state=10)
xgboost.fit(x_treino, y_treino)
xgboost.score(x_treino, y_treino)       # Determination coefficient

### TEST
xgboost.score(x_teste, y_teste)
previsoes_teste = xgboost.predict(x_teste)

### METRICS
MAE_XGB = abs(y_teste - previsoes_teste).mean()
meanMAE_XGB = mean_absolute_error(y_teste, previsoes_teste)
MSE_XGB = mean_squared_error(y_teste, previsoes_teste)
RMSE_XGB = np.sqrt(mean_squared_error(y_teste, previsoes_teste))

### CROSS VALIDATION
from xgboost import XGBRegressor
modelo = XGBRegressor(n_estimators=180, max_depth=3, learning_rate=0.05, objective = "reg:squarederror")
resultado = cross_val_score(modelo, indep, Ery, cv = kfold)
resultado_XGB = resultado

print("Average Coefficient of Determination: %.2f%%" % (resultado.mean() * 100.0))

########################################
# REGRESSION WITH LIGHTGBM
########################################
#pip install lightgbm
import lightgbm as lgb
lgbm = lgb.LGBMRegressor(num_leaves=50, max_depth=3, learning_rate=0.1, n_estimators=50, random_state=10)
lgbm.fit(x_treino, y_treino)
lgbm.score(x_treino, y_treino)      # Determination coefficient

### TEST
lgbm.score(x_teste, y_teste)
previsoes_teste = lgbm.predict(x_teste)

### METRICS
MAE_GBM = abs(y_teste - previsoes_teste).mean()
meanMAE_GBM = mean_absolute_error(y_teste, previsoes_teste)
MSE_GBM = mean_squared_error(y_teste, previsoes_teste)
RMSE_GBM = np.sqrt(mean_squared_error(y_teste, previsoes_teste))

### CROSS VALIDATION
modelo = lgb.LGBMRegressor(num_leaves=50, max_depth=3, learning_rate=0.1, n_estimators=50)
resultado = cross_val_score(modelo, indep, Ery, cv = kfold)
resultado_LGB = resultado

print("Average Coefficient of Determination: %.2f%%" % (resultado.mean() * 100.0))

########################################
# REGRESSION WITH CATBOOST
########################################
#pip install catboost
from catboost.core import CatBoostRegressor
catboost = CatBoostRegressor (iterations=100, learning_rate=0.08, depth = 5, random_state = 10)
catboost.fit(x_treino, y_treino)
catboost.score(x_treino, y_treino)      # Determination coefficient

### TEST
catboost.score(x_teste, y_teste)
previsoes_teste = catboost.predict(x_teste)

### METRICS
MAE_CAT = abs(y_teste - previsoes_teste).mean()
meanMAE_CAT = mean_absolute_error(y_teste, previsoes_teste)
MSE_CAT = mean_squared_error(y_teste, previsoes_teste)
RMSE_CAT = np.sqrt(mean_squared_error(y_teste, previsoes_teste))

### CROSS VALIDATION
modelo = CatBoostRegressor (iterations=100, learning_rate=0.08, depth = 5, random_state = 10)
resultado = cross_val_score(modelo, indep, Ery, cv = kfold)
resultado_CAT = resultado

print("Average Coefficient of Determination: %.2f%%" % (resultado.mean() * 100.0))

### Saving data to Deploy
np.savetxt('x_indep_hires.csv', indep, delimiter=',')
np.savetxt('y_dep_hires.csv', Ery, delimiter=',')

### Saving the information
data = np.array([resultado_RLM,resultado_SVR,resultado_AD, resultado_RF, resultado_XGB, resultado_LGB, resultado_CAT]).T
df = pd.DataFrame(data, columns=['RLM', 'SVM', 'AD', 'RF', 'XGB', 'GBM','CAT'])
df.to_excel(excel_writer = "coef_det_hires.xlsx")

metricas = np.array([[MAE_RLM,meanMAE_RLM,MSE_RLM,RMSE_RLM],[MAE_SVM,meanMAE_SVM,MSE_SVM,RMSE_SVM],[MAE_AD,meanMAE_AD,MSE_AD,RMSE_AD],
                     [MAE_RF,meanMAE_RF,MSE_RF,RMSE_RF],[MAE_XGB,meanMAE_XGB,MSE_XGB,RMSE_XGB],[MAE_GBM,meanMAE_GBM,MSE_GBM,RMSE_GBM],
                     [MAE_CAT,meanMAE_CAT,MSE_CAT,RMSE_CAT]]).T
dfmet = pd.DataFrame(metricas, columns=['RLM', 'SVM', 'AD', 'RF', 'XGB', 'GBM','CAT'])
dfmet.to_excel(excel_writer = "Errors_hires.xlsx")