
# importing the necessary dependencies
from flask import Flask, render_template, request,send_file,jsonify
from flask_cors import CORS,cross_origin
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
sns.set()
import pickle

app = Flask(__name__) # initializing a flask app

@app.route('/',methods=['GET'])  # route to display the home page
@cross_origin()
def homePage():
    return render_template("index.html")

@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
@cross_origin()
def index():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            fixed_acidity=float(request.form['fixed_acidity'])
            volatile_acidity = float(request.form['volatile_acidity'])
            citric_acid = float(request.form['citric_acid'])
            residual_sugar = float(request.form['residual_sugar'])
            chlorides = float(request.form['chlorides'])
            free_sulfur_dioxide = float(request.form['free_sulfur_dioxide'])
            total_sulfur_dioxide = float(request.form['total_sulfur_dioxide'])
            density = float(request.form['density'])
            pH = float(request.form['pH'])
            sulphates = float(request.form['sulphates'])
            alcohol = float(request.form['alcohol'])


            filename = 'modelForPrediction_1.sav'
            loaded_model = pickle.load(open(filename, 'rb')) # loading the model file from the storage

            #loading Scaler pickle file
            scaler = pickle.load(open('standardScalar_1.sav', 'rb'))

            # predictions using the loaded model file and scaler file
            prediction = loaded_model.predict(scaler.transform([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]]))


            print('prediction is', prediction)
            # showing the prediction results in a UI
            if prediction == 1:
                return render_template('quality_1.html')
            elif prediction == 2:
                return render_template('quality_2.html')
            elif prediction == 3:
                return render_template('quality_3.html')
            elif prediction == 4:
                return render_template('quality_4.html')
            elif prediction == 5:
                return render_template('quality_5.html')
            elif prediction == 6:
                return render_template('quality_6.html')
            elif prediction == 7:
                return render_template('quality_7.html')
            elif prediction == 8:
                return render_template('quality_8.html')
            elif prediction == 9:
                return render_template('quality_9.html')
            elif prediction == 10:
                return render_template('quality_10.html')

            else:
                return "something is wrong"

        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'

    else:
        return render_template('index.html')



@app.route('/csv',methods=['POST','GET']) # route to show the predictions in a web UI
@cross_origin()
def csv():
    if request.method == 'POST':
        try:
            #reading csv file
            uploaded_file = request.files['upload_file']
            filename = uploaded_file.filename

            #procede only if file is available
            if uploaded_file.filename != '':
                uploaded_file.save(filename)
                data = pd.read_csv(filename)


                # procede only if file is in correct format
                if len(data.columns) == 11:

                    #filling NaN values if present in dataset
                    data['fixed acidity'].fillna(value=round(data['fixed acidity'].mean()), inplace=True)
                    data['volatile acidity'].fillna(value=round(data['volatile acidity'].mean()), inplace=True)
                    data['citric acid'].fillna(value=round(data['citric acid'].mean()), inplace=True)
                    data['residual sugar'].fillna(value=round(data['residual sugar'].mean()), inplace=True)
                    data['chlorides'].fillna(value=round(data['chlorides'].mean()), inplace=True)
                    data['free sulfur dioxide'].fillna(value=data['free sulfur dioxide'].mean(), inplace=True)
                    data['total sulfur dioxide'].fillna(value=data['total sulfur dioxide'].mean(), inplace=True)
                    data['density'].fillna(value=round(data['density'].mean()), inplace=True)
                    data['pH'].fillna(value=round(data['pH'].mean()), inplace=True)
                    data['sulphates'].fillna(value=round(data['sulphates'].mean()), inplace=True)
                    data['alcohol'].fillna(value=round(data['alcohol'].mean()), inplace=True)

                    # loading the model file from the storage
                    model_filename = 'modelForPrediction_1.sav'
                    loaded_model = pickle.load(open(model_filename, 'rb'))

                    # loading Scaler pickle file
                    scaler = pickle.load(open('standardScalar_1.sav', 'rb'))


                    #deleting previous files present in csv_file folder
                    csv_files = './csv_file'
                    list_of_files = os.listdir(csv_files)
                    for csfile in list_of_files:
                        try:
                            os.remove("./csv_file/" + csfile)
                        except Exception as e:
                            print('error in deleting:  ', e)

                    # making prediction
                    prediction = loaded_model.predict(scaler.transform(data))
                    data['Prediction Of Quality Of Wine'] = prediction

                    #saving pandas dataframe as a csv file in csv_file folder
                    result_file = './csv_file/result_output_data.csv'
                    data.to_csv(result_file)

                    #plot for prediction analysis
                    sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})
                    total_pridiction = sns.catplot(x='Prediction Of Quality Of Wine', kind='count', data=data)


                    # deleting previous graph images present in statistics folder
                    image_files = './static/statistics'
                    list_of_files = os.listdir(image_files)
                    for imgfile in list_of_files:
                        try:
                            os.remove("./static/statistics/" + imgfile)
                        except Exception as e:
                            print('error in deleting:  ', e)

                    #save graph in statictics folder inside static
                    output_path_total = './static/statistics/output_prediction.png'
                    total_pridiction.savefig(output_path_total)


                    return render_template('csv.html')

                else:
                    return 'Error: Please Make Sure that csv file is in standard acceptable format,Please go through given Sample csv file format'


            else:
                return 'File Not Found'


        except Exception as e:
            print('The Exception message is: ', e)
            return 'something is wrong'

    else:
        return render_template('index.html')



@app.route('/uploadfile',methods=['POST','GET'])  #
@cross_origin()
def uploadfile():
    return render_template('upload.html')



@app.route('/download')  #
@cross_origin()
def download_file():
    p = './csv_file/result_output_data.csv'
    return send_file(p, as_attachment=True)



@app.route('/statistics',methods=['POST','GET'])  #
@cross_origin()
def stat_graph():
    return render_template('show_statistics.html')





if __name__ == "__main__":
    #to run locally
    #app.run(host='127.0.0.1', port=8000, debug=True)

    #to run on cloud
	app.run(debug=True) # running the app