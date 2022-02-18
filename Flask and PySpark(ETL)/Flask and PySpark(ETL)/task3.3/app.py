from flask import Flask, request
from flask.templating import render_template
import numpy as np
import pandas as pd
from pandas import DataFrame
import pickle

app = Flask(__name__)

# prediction function
def ValuePredictor(data):
    #print(data)

    to_predict = np.array(to_predict_list).reshape(1, 23)
   
    loaded_model = pickle.load(open("model.pkl", "rb"))
    result = loaded_model.predict(to_predict)
    return result[0]
 
@app.route('/')
def index():
    return render_template('index1.html')
    
@app.route('/result', methods = ['POST'])
def result():   
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        print(to_predict_list)
        column_names=['age','job','marital','education','default','housing','loan','contact','month','day_of_week','duration',
        'campaign','pdays','previous','poutcome','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed',
        'State_Name_x','City_Name','Region_Name']
        data=pd.DataFrame(to_predict_list,index=column_names).T
        print(data)
        result = ValuePredictor(data)             
        return render_template("results.html", prediction = result)
        #return to_predict_list


if __name__ == "__main__":
    app.run(debug=True, port=8000)