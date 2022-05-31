import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

dataset = pd.read_csv('blindchess_dataset.csv')

dataset_X = dataset.iloc[:,[0]].values

#from sklearn.preprocessing import MinMaxScaler
#sc = MinMaxScaler(feature_range = (0,1))
#dataset_scaled = sc.fit_transform(dataset_X)

new_dataset = dataset
new_dataset[["Moves", "Rating"]] = new_dataset[["Moves", "Rating"]].replace(0, np.NaN) 
new_dataset.isnull().sum()
new_dataset["Moves"].fillna(new_dataset["Moves"].mean(), inplace = True)
new_dataset["Rating"].fillna(new_dataset["Rating"].mean(), inplace = True)
convert_dict = {'Moves': int,
                'Rating': int
               }
  
new_dataset = new_dataset.astype(convert_dict)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    float_features = request.form['Moves']
    final_features = float_features
    prediction = model.predict( [[final_features]] )

    #if prediction == 1:
        #pred = "You have Diabetes, please consult a Doctor."
    #elif prediction == 0:
        #pred = "You don't have Diabetes."
    output = prediction

    return render_template('index.html', prediction_text='{}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
