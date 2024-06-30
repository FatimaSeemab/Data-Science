from flask import Flask,jsonify,request
import utils

app=Flask(__name__)
@app.route('/get_location_names', methods=['GET'])
def get_location_name():
    response=jsonify(
        {'locations':utils.get_location_names()}
    )
    response.headers.add("Access-Control-Allow-Origin","*")
    return response

@app.route('/predict_home_price',methods=["POST"])
def predict_home_price():
    total_sqft = request.form['total_sqft']
    location = request.form['location']
    bhk = int(request.form['bhk'])
    bath = int(request.form['bath'])

    response=jsonify({
        'estimated_price':utils.get_estimated_price(location,total_sqft,bhk,bath)
    })
    response.headers.add("Access-Control-Allow-Origin","*")
    return response

if __name__ == '__main__':
    print("Starting Python Flask Server For Home Price Prediction...")
    utils.load_saved_artifacts()
    app.run(port=5000,debug=True)
