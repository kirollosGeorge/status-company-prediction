from flask import Flask, render_template, request
from dummies_dic.dummies import Is_Closed_dummies , Category_Code_dummies,country_code_dummies
import joblib

app = Flask(__name__)

# import the model and his scaler
scaler = joblib.load('Machine learning Model/scaler.h5')
RF = joblib.load('Machine learning Model/Rf_selected_91.h5')
QDA = joblib.load('Machine learning Model/QDA.h5')


@app.route('/')
def index():
    return render_template('index.html')




@app.route('/home')
def home():
    all_data = request.args
    Category_Code = Category_Code_dummies[all_data["Category Code"]]
    country_code = country_code_dummies[all_data['country_code']]
    Is_Closed = Is_Closed_dummies[all_data['Is_Closed']]
    Age_Day = float(all_data['Age_Day']) # 
    founded_at = float(all_data['founded_at'])
    Funding_total_usd = float(all_data['Funding_total_usd'])#
    last_funding_at = float(all_data['last_funding_at'])#
    data =Is_Closed + [ Age_Day, founded_at,Funding_total_usd,last_funding_at] + Category_Code + country_code
    data = scaler.transform([ data])
    pred = round(QDA.predict(data)[0])
    #return render_template('prediction.html' , pred =pred ) 
   
    if pred == 0 :
        pred='Operating'
        return render_template('prediction.html' , pred =pred )
    else:
        pred2 = round(RF.predict(data)[0])
        if pred2 == 0:
            pred='Operating'
            return render_template('prediction.html' , pred = pred)
        elif pred2 == 1 :
            pred='acquired'
            return render_template('prediction.html' , pred = pred)
        elif pred2 == 2 :
            pred='closed'
            return render_template('prediction.html' , pred = pred)
        elif pred2 == 3 :
            pred='ipo'
            return render_template('prediction.html' , pred = pred)    


    



if __name__ == "__main__":
    app.run('127.0.0.1', port = 5500)
'''
 all_data = request.args
    name = name_dummies[all_data["name"]]
    car_tyb = car_type[all_data['Car type']]
    dest = destination_dummies[all_data['Destination']]
    source = source_dummies[all_data['source']]
    product = product_id_dummies[all_data['Product ID']]
    suger = Suger_dummies[all_data['Suger']]
    UV = Uv_dummies[all_data['UV']]
    Month = Month_dummies[all_data['Month']]
    Icon = icon_dummies[all_data['Icon']]
    Distance = float(all_data['Distance'])
    data = Month + suger + UV + [Distance] + source + dest + product + name + Icon + car_tyb 
    data = scaler.transform([data])
    pred = round(model.predict(data)[0])
'''