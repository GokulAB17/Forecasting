#Forecast the CocaCola prices and Airlines Passengers data set. Prepare a document for each model explaining 
#how many dummy variables you have created and RMSE value for each model. Finally which model you will use for 
#Forecasting.

#Import Dataset
import pandas as pd
Coca= pd.read_excel(r"CocaCola_Sales_Rawdata.xlsx")
Coca.head()

#EDA and Data Preprocessing
quarter =["Q1","Q2","Q3","Q4"] 
import numpy as np
p = Coca["Quarter"][0]
p[0:2]
Coca['quarter']= 0

for i in range(42):
    p = Coca["Quarter"][i]
    Coca['quarter'][i]= p[0:2]
    
quarter_dummies = pd.DataFrame(pd.get_dummies(Coca['quarter']))

Coca1= pd.concat([Coca,quarter_dummies],axis = 1)

Coca1["t"] = np.arange(1,43)

Coca1["t_squared"] = Coca1["t"]*Coca1["t"]
Coca1.columns
Coca1["log_Sales"] = np.log(Coca1["Sales"])

#Plotting graph for Sales data 
Coca1.Sales.plot()

#from graph trend is upward, additive sesonality


#Splitting Data into train and test as per horizon 
Train = Coca1.head(38)
Test = Coca1.tail(4)


####################### L I N E A R ##########################
import statsmodels.formula.api as smf 

linear_model = smf.ols('Sales~t',data=Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_linear))**2))
rmse_linear
#591.55
##################### Exponential ##############################

Exp = smf.ols('log_Sales~t',data=Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test["Sales"])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp
#466.247
#################### Quadratic ###############################

Quad = smf.ols('Sales~t+t_squared',data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_squared"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_Quad))**2))
rmse_Quad
#475.56
################### Additive seasonality ########################

add_sea = smf.ols('Sales~Q1+Q2+Q3+Q4',data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[["Q1","Q2","Q3","Q4"]]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea))**2))
rmse_add_sea
#1860.02
################## Additive Seasonality Quadratic ############################

add_sea_Quad = smf.ols('Sales~t+t_squared+Q1+Q2+Q3+Q4',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[["Q1","Q2","Q3","Q4",'t','t_squared']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad 
#301.73
################## Multiplicative Seasonality ##################

Mul_sea = smf.ols('log_Sales~Q1+Q2+Q3+Q4',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea
#1963.38
##################Multiplicative Additive Seasonality ###########

Mul_Add_sea = smf.ols('log_Sales~t+Q1+Q2+Q3+Q4',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea 
#225.52


#Tabulating all RMSE values for each model
data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
table_rmse=pd.DataFrame(data)
table_rmse
#               MODEL  RMSE_Values
#0        rmse_linear   591.553296
#1           rmse_Exp   466.247973
#2          rmse_Quad   475.561835
#3       rmse_add_sea  1860.023815
#4  rmse_add_sea_quad   301.738007
#5      rmse_Mult_sea  1963.389640
#6  rmse_Mult_add_sea   225.524390

#Storing as Document as csv file
table_rmse.to_csv(r"CocaCola_RMSE.csv", encoding="UTF-8")

#Storing data and dummies prepared as csv file
Coca1.to_csv(r"CocaColaDummies.csv", encoding="UTF-8")

#we can se rmse value for multiplicative additive seasonality has least rmse value 
#we choose that model as our final model for prediction