#Forecast the CocaCola prices and Airlines Passengers data set. Prepare a document for each model explaining 
#how many dummy variables you have created and RMSE value for each model. Finally which model you will use for 
#Forecasting.

#Import Dataset
import pandas as pd
ALdata= pd.read_excel(r"Airlines+Data.xlsx")
ALdata.head()


#EDA and Data Preprocessing
ALdata["Month"] = ALdata.Month.dt.strftime("%b-%Y")

ALdata.rename(columns={"Passengers":"P"},inplace=True)

month =['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'] 
import numpy as np
p = ALdata["Month"][0]
p[0:3]
ALdata['months']= 0

for i in range(96):
    p = ALdata["Month"][i]
    ALdata['months'][i]= p[0:3]

month_dummies = pd.DataFrame(pd.get_dummies(ALdata['months']))
ALdata1 = pd.concat([ALdata,month_dummies],axis = 1)

ALdata1["t"] = np.arange(1,97)

ALdata1["t_squared"] = ALdata1["t"]*ALdata1["t"]
ALdata1.columns
ALdata1["log_P"] = np.log(ALdata1["P"])

#Plotting graph for Passengers data
ALdata1.P.plot()
#we can see from graph trend is upward season is multiplicative


#Splitting Data into train and test as per horizon
Train = ALdata1.head(84)
Test = ALdata1.tail(12)

# to change the index value in pandas data frame 
# Test.set_index(np.arange(1,13))

####################### L I N E A R ##########################
import statsmodels.formula.api as smf 

linear_model = smf.ols('P~t',data=Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['P'])-np.array(pred_linear))**2))
rmse_linear
#53.199
##################### Exponential ##############################

Exp = smf.ols('log_P~t',data=Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test["P"])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp
#46.05
#################### Quadratic ###############################

Quad = smf.ols('P~t+t_squared',data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_squared"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['P'])-np.array(pred_Quad))**2))
rmse_Quad
#48.05
################### Additive seasonality ########################

add_sea = smf.ols('P~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['P'])-np.array(pred_add_sea))**2))
rmse_add_sea
#132.82
################## Additive Seasonality Quadratic ############################

add_sea_Quad = smf.ols('P~t+t_squared+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','t','t_squared']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['P'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad 
#26.36
################## Multiplicative Seasonality ##################

Mul_sea = smf.ols('log_P~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['P'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea
#140.06
##################Multiplicative Additive Seasonality ###########

Mul_Add_sea = smf.ols('log_P~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['P'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea 
#10.5191

#Tabulating all RMSE values for each model
data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
table_rmse=pd.DataFrame(data)
table_rmse
#               MODEL  RMSE_Values
#0        rmse_linear    53.199237
#1           rmse_Exp    46.057361
#2          rmse_Quad    48.051889
#3       rmse_add_sea   132.819785
#4  rmse_add_sea_quad    26.360818
#5      rmse_Mult_sea   140.063202
#6  rmse_Mult_add_sea    10.519173

#Storing as Document as csv file
table_rmse.to_csv(r"AirLinesRMSE.csv", encoding="UTF-8")


#Storing data and dummies prepared as csv file
ALdata1.to_csv(r"AirLinesDummies.csv", encoding="UTF-8")

#we can se rmse value for multiplicative additive seasonality has least rmse value 
#we choose that model as our final model for prediction