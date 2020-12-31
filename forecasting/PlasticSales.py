#Forecast the CocaCola prices and Airlines Passengers data set. Prepare a document for each model explaining 
#how many dummy variables you have created and RMSE value for each model. Finally which model you will use for 
#Forecasting.
#Import Dataset
import pandas as pd
Plas= pd.read_csv(r"PlasticSales.csv",encoding="UTF-8")
Plas.head()

#EDA and Data Preprocessing
month =['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'] 
import numpy as np
p = Plas["Month"][0]
p[0:3]
Plas['months']= 0

for i in range(0,60,1):
    p = Plas["Month"][i]
    Plas['months'][i]= p[0:3]
    
month_dummies = pd.DataFrame(pd.get_dummies(Plas['months']))
Plas1 = pd.concat([Plas,month_dummies],axis = 1)

Plas1["t"] = np.arange(1,61)

Plas1["t_squared"] = Plas1["t"]*Plas1["t"]
Plas1.columns
Plas1["log_Sales"] = np.log(Plas1["Sales"])

#Plotting graph for Sales data 
Plas1.Sales.plot()
#from graph trend is upward, additive sesonality

#Splitting Data into train and test as per horizon 
Train = Plas1.head(48)
Test = Plas1.tail(12)

#Models Building
####################### L I N E A R ##########################
import statsmodels.formula.api as smf 

linear_model = smf.ols('Sales~t',data=Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_linear))**2))
rmse_linear
#260.9378
##################### Exponential ##############################

Exp = smf.ols('log_Sales~t',data=Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test["Sales"])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp
#268.6938
#################### Quadratic ###############################

Quad = smf.ols('Sales~t+t_squared',data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_squared"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_Quad))**2))
rmse_Quad
#297.406
################### Additive seasonality ########################

add_sea = smf.ols('Sales~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea))**2))
rmse_add_sea
#235.60
################## Additive Seasonality Quadratic ############################

add_sea_Quad = smf.ols('Sales~t+t_squared+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','t','t_squared']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad 
#218.19
################## Multiplicative Seasonality ##################

Mul_sea = smf.ols('log_Sales~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea
#239.654
##################Multiplicative Additive Seasonality ###########

Mul_Add_sea = smf.ols('log_Sales~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea 
#160.68

#Tabulating all RMSE values for each model
data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
table_rmse=pd.DataFrame(data)

#Storing as Document as csv file
table_rmse.to_csv(r"PSalesRMSE.csv", encoding="UTF-8")

#Storing data and dummies prepared as csv file
Plas1.to_csv(r"PSalesDummies.csv", encoding="UTF-8")
#               MODEL  RMSE_Values
#0        rmse_linear   260.937814
#1           rmse_Exp   268.693839
#2          rmse_Quad   297.406710
#3       rmse_add_sea   235.602674
#4  rmse_add_sea_quad   218.193876
#5      rmse_Mult_sea   239.654321
#6  rmse_Mult_add_sea   160.683329


#we can se rmse value for multiplicative additive seasonality has least rmse value 
#we choose that model as our final model for prediction