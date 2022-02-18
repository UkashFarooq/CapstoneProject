from flask import Flask, request
from flask.templating import render_template
import numpy as np
import pandas as pd
from pandas import DataFrame
import pickle
from variables import *

app = Flask(__name__)

cat=[ 'job', 'marital', 'education', 'default', 'housing', 'loan',
'contact', 'month', 'day_of_week', 'poutcome', 'Region_Name']
num=['age','duration','campaign','pdays','previous','emp.var.rate', 'cons.price.idx',
'cons.conf.idx', 'euribor3m', 'nr.employed']



def switchnum(x,col):
    if col=='age':
        x=(x-agem)/agestd
    elif col=='campaign':
        x=(x-campaignm)/campaignstd
    elif col=='previous':
        x=(x-previousm)/previousstd 
    elif col=='emp.var.rate':
        x=(x-empmean)/empstd
    elif col=='cons.conf.idx':
        x=(x-consconfmean)/consconfstd
    elif col=='cons.price.idx':
        x=(x-conspricemean)/conspricestd
    elif col=='euribor3m':
        x=(x-euribormean)/euriborstd
    elif col=='nr.employed':
        x=(x-emplomean)/emplostd
    elif col=='duration':
        if 0<x<200:
            x=1
        elif 200<=x<700:
            x=2
        else:
            x=0
    elif col=='pdays':
        if x==999:
            x=0
        else:
            x=1
    return x

dictionary={'yes':1,'no':0,'unknown':-1}
def switchcat(x,col):
    if col=='job':
        x=job[x]
    elif col=='marital':
        x=marital[x]
    elif col=='education':
        lst=['basic.9y','basic.6y','basic.4y']  
        if x in lst:
            x='middle.school'
        x=education[x]
    elif col=='default':
        x=dictionary[x]
    elif col=='housing':
        x=dictionary[x]
    elif col=='loan':
        x=dictionary[x]
    elif col=='contact':
        x=contact[x]
    elif col=='month':
        month_dict={'may':5,'jul':7,'aug':8,'jun':6,'nov':11,'apr':4,'oct':10,'sep':9,'mar':3,'dec':12}
        x=month_dict[x]
    elif col=='day_of_week':
        day_dict={'thu':5,'mon':2,'wed':4,'tue':3,'fri':6}
        x=day_dict[x]
    elif col=='poutcome':
        x=poutcome[x]
    elif col=='Region_Name':
        x=region[x]
    return x

def getdata(data):
    liss=[]
    for col in data.columns:
        if col in cat:
            ld=switchcat(data.loc[0][col],col)
            liss.append(ld)
        elif col in num:
            n=float(data.loc[0][col])
            scd=switchnum(n,col)
            liss.append(scd)
    return liss


def ValuePredictor(data):
    to_predict_list=getdata(data)
    to_predict = np.array(to_predict_list).reshape(1, 21)
   
    loaded_model = pickle.load(open("model.pkl", "rb"))
    result = loaded_model.predict(to_predict)
    print('Result:',result)
    if result[0]==0:
        return 'No - the customer will not opt for the term deposit'
    else:
        return 'Yes - the customer will opt for the term deposit'
 
@app.route('/')
def index():
    return render_template('index.html')
    
@app.route('/result', methods = ['POST'])
def result():   
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        print(to_predict_list)
        column_names=['age','duration','campaign','pdays','job','education','marital','default',
        'housing','loan','contact','month','previous','emp.var.rate','cons.price.idx',
        'cons.conf.idx','euribor3m','nr.employed','Region_Name','day_of_week','poutcome']
        data=pd.DataFrame(to_predict_list,index=column_names).T
        print(data)
        result = ValuePredictor(data)             
        return render_template("results.html", prediction = result)

if __name__ == "__main__":
    app.run(debug=True, port=8000)