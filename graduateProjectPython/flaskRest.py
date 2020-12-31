from flask import Flask
from flask_jsonpify import jsonpify
from flask_restful import reqparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
import json


import numpy as np
import pandas as pd



app = Flask(__name__)
def Test():
    train = pd.read_csv('./csvfile/cardio.csv')
    
    train["plus"] = train["smoke"]*train["alco"]
    
    train["age_year"] = round(train["age_year"],0).astype(np.int64)
    train["BMI"] = round(train["weight"]/(train["height"]*train["height"]/10000),2).astype(np.float64)
    train = train.dropna(0)
    
    
    train = train[(train.BMI <= 50) & (train.BMI >= 10)]
    y = train["cardio"]
    print(train.shape)
    print(y.shape)
    train = train.drop(["id","age_days","cardio"],1)
    print(train.shape)
    
    
    print(train["BMI"].max())
    print(train["BMI"].min())
    
    
    train.loc[(train["ap_hi"]>=140) & (train["ap_hi"] < 200),'ap_hi'] = 3 #high 
    train.loc[(train["ap_hi"] < 90 ) & (train["ap_hi"] >= 60),'ap_hi'] = 1 #low
    train.loc[(train["ap_hi"] < 140 ) & (train["ap_hi"] >= 90) ,'ap_hi'] = 2 #normal
    
    
    train = train.drop(["weight","height","ap_lo"],1)
    rf = LGBMClassifier(n_estimators = 200 ,num_leaves = 25,  colsample_bytree = 0.6, subsample = 0.6 )
    
    
    xf_train,xf_test,yf_train,yf_test = train_test_split(train,y,test_size = 0.2,random_state = 1)
    
    rf.fit(xf_train,yf_train)
    
    odd = round(rf.score(xf_test, yf_test)*100,1)
    print(odd)
    
    print(train.shape)
    importances_df = pd.DataFrame(rf.feature_importances_).rename({0:"importances"},axis = 1)
    importances_df["columns"] = xf_train.columns 
    importances_df = importances_df.sort_values("importances",ascending = False) 
    importances_df["importances"] = (importances_df["importances"]/importances_df["importances"].values.sum())*100
    print(importances_df.head(10))
    
#Test()

@app.route('/', methods=['GET'])
def start():
    return "test OK2"
    

#87
@app.route('/heartDisease', methods=['GET'])
def heartDisease():
    
    train = pd.read_csv('./csvfile/framingham.csv')
    y = train["TenYearCHD"]
    train = train.drop(["TenYearCHD","education","totChol","sysBP","diaBP","heartRate","glucose"],1)
    rf = LGBMClassifier(n_estimators = 100 ,num_leaves = 15 ,  min_child_samples = 10, colsample_bytree = 0.6, subsample = 0.6 )
    rf.fit(train,y)
    
    parser = reqparse.RequestParser()
    parser.add_argument('BMI', type=float)
    parser.add_argument('age', type=int)
    parser.add_argument('cigsPerDay', type=int)
    parser.add_argument('prevalentHyp', type=int)
    parser.add_argument('male', type=int)
    parser.add_argument('currentSmoker', type=int)
    parser.add_argument('BPMeds', type=int)
    parser.add_argument('diabetes', type=int)
    parser.add_argument('prevalentStroke', type=int)
    
    args = parser.parse_args()
    
    
    data = {'BMI': [args['BMI']], 'age' : [args['age']] ,'cigsPerDay': [args['cigsPerDay']] , 'prevalentHyp': [args['prevalentHyp']] ,
            'male' : [args['male']] , 'currentSmoker' : [args['currentSmoker']] ,'BPMeds' : [args['BPMeds']],
            'diabetes' : [args['diabetes']],'prevalentStroke' : [args['prevalentStroke']]}
    x_r = pd.DataFrame(data, columns = ['BMI' , 'age' , 'cigsPerDay' , 'prevalentHyp' , 'male' , 'currentSmoker' ,'BPMeds','diabetes' ,'prevalentStroke'] ,index = [1])
    
    pred = rf.predict(x_r)
    print(pred)
    proba = rf.predict_proba(x_r)
    
    if(pred == 0):
        odd = (1 - proba[0][0])*100
    else:
        odd = proba[0][1]*100
    print(odd)
    
  
    data = {
        "result" : pred.tolist(),
        "odd" : round(odd,2)
        }
    s1 = json.dumps(data)
    res = json.loads(s1)
    return jsonpify( np.array(res).tolist())
    

#80.1
@app.route('/diabetes', methods=['GET'])
def diabetes():
    
    train = pd.read_csv('./csvfile/diabetes2.csv')
    y = train["Outcome"]
    train = train.drop(["Outcome","DiabetesPedigreeFunction"],1)
    rf = LGBMClassifier(n_estimators = 125, num_leaves =15)
    rf.fit(train,y)
    
    parser = reqparse.RequestParser()
    parser.add_argument('Pregnancies', type=int)
    parser.add_argument('Glucose', type=int)
    parser.add_argument('BloodPressure', type=int)
    parser.add_argument('SkinThickness', type=int)
    parser.add_argument('Insulin', type=int)
    parser.add_argument('BMI', type=float)
    parser.add_argument('Age', type=int)
    
    args = parser.parse_args()
    
    
    data = {'Pregnancies': [args['Pregnancies']], 'Glucose' : [args['Glucose']] ,'BloodPressure': [args['BloodPressure']] , 'SkinThickness': [args['SkinThickness']] ,
            'Insulin' : [args['Insulin']] , 'BMI' : [args['BMI']] ,'Age' : [args['Age']]}
    x_r = pd.DataFrame(data, columns = ['Pregnancies' , 'Glucose' , 'BloodPressure' , 'SkinThickness' , 'Insulin' , 'BMI' ,'Age'] ,index = [1])
    
    pred = rf.predict(x_r)
    print(pred)
    proba = rf.predict_proba(x_r)
    
    
    if(pred == 0):
        odd = (1 - proba[0][0])*100
    else:
        odd = proba[0][1]*100
    print(odd)
    
  
    data = {
        "result" : pred.tolist(),
        "odd" : round(odd,2)
        }
    s1 = json.dumps(data)
    res = json.loads(s1)
    return jsonpify( np.array(res).tolist())
   

#72.6
@app.route('/heartDisease2', methods=['GET'])
def heartDisease2():
    train = pd.read_csv('./csvfile/cardio.csv')
    
    train["plus"] = train["smoke"]*train["alco"]
    
    train["age_year"] = round(train["age_year"],0).astype(np.int64)
    train["BMI"] = round(train["weight"]/(train["height"]*train["height"]/10000),2).astype(np.float64)
    train = train.dropna(0)
    
    
    train = train[(train.BMI <= 50) & (train.BMI >= 10)]
    y = train["cardio"]
    print(train.shape)
    print(y.shape)
    train = train.drop(["id","age_days","cardio"],1)
    print(train.shape)
    
    
    print(train["BMI"].max())
    print(train["BMI"].min())
    
    
    train.loc[(train["ap_hi"]>=140) & (train["ap_hi"] < 200),'ap_hi'] = 3 #high 
    train.loc[(train["ap_hi"] < 90 ) & (train["ap_hi"] >= 60),'ap_hi'] = 1 #low
    train.loc[(train["ap_hi"] < 140 ) & (train["ap_hi"] >= 90) ,'ap_hi'] = 2 #normal
    
    
    train = train.drop(["weight","height","ap_lo"],1)
    rf = LGBMClassifier(n_estimators = 200 ,num_leaves = 25,  colsample_bytree = 0.6, subsample = 0.6 )
    
    rf.fit(train,y)
    importances_df = pd.DataFrame(rf.feature_importances_).rename({0:"importances"},axis = 1)
    importances_df["columns"] = train.columns 
    importances_df = importances_df.sort_values("importances",ascending = False) 
    importances_df["importances"] = (importances_df["importances"]/importances_df["importances"].values.sum())*100
    print(importances_df.head(10))
    
    
    parser = reqparse.RequestParser()
    parser.add_argument('Sex', type=int)
    parser.add_argument('BloodPressure', type=int)
    parser.add_argument('Cholesterol', type=int)
    parser.add_argument('Smoke', type=int)
    parser.add_argument('Alchol', type=int)
    parser.add_argument('Active', type=int)
    parser.add_argument('BMI', type=float)
    parser.add_argument('Age', type=int)
    parser.add_argument('Glucose', type=int)
    
    args = parser.parse_args()
    
    
    data = {'gender': [args['Sex']], 'ap_hi' : [args['BloodPressure']] ,'cholesterol': [args['Cholesterol']] , 'smoke': [args['Smoke']] ,
            'alco' : [args['Alchol']] , 'active' : [args['Active']] ,'BMI' : [args['BMI']] ,'age_year' : [args['Age']] 
            ,'gluc' : [args['Glucose']] ,'plus' : [args['Smoke'] * args['Alchol']] }
    x_r = pd.DataFrame(data, columns = ['gender' , 'ap_hi' , 'cholesterol' , 'smoke' , 'alco' , 'active' ,'BMI', 'age_year' ,'gluc' ,'plus'] ,index = [1])
    
    pred = rf.predict(x_r)
    print(pred)
    proba = rf.predict_proba(x_r)
    
    print(x_r)
    
    
    if(pred == 0):
        odd = (1 - proba[0][0])*100
    else:
        odd = proba[0][1]*100
    print(odd)
    
  
    data = {
        "result" : pred.tolist(),
        "odd" : round(odd,2)
        }
    s1 = json.dumps(data)
    res = json.loads(s1)
    return jsonpify( np.array(res).tolist())
    
    
    
if __name__=='__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
    
    
    
    
    
