import json
import joblib
import numpy as np

locations=None
model=None
data_columns=None
def get_location_names():
    return locations

def get_estimated_price(location,sqft,bhk,bath):
        global model
        try:
            loc_index = data_columns.index(location.lower)
        except:
            loc_index=-1
        x = np.zeros(len(data_columns))
        x[0] = sqft
        x[1] = bath
        x[2] = bhk
        if loc_index >= 0:
            x[loc_index] = 1
        return model.predict([x])[0]

def load_saved_artifacts():
    print("loading saved artifacts")
    global data_columns
    global locations
    global model

    with open('/Users/humashehwana/Documents/DataScience/RoadmapProject/Project1/model/columns.json') as f:
        data_columns=json.load(f)['data_columns']
        locations=data_columns[3:]

    with open('/Users/humashehwana/Documents/DataScience/RoadmapProject/Project1/model/linear_regression.pkl','rb') as f:
        model=joblib.load(f)

    print("loading saved artifacts done")

if __name__=="__main__":
    load_saved_artifacts()
    print(get_location_names())
    print(get_estimated_price('1st Phase JP Nagar',1000,3,3))
    print(get_estimated_price('1st Phase JP Nagar',1000,2,2))
    print(get_estimated_price('Kalhalli',1000,2,2))
    print(get_estimated_price('Ejipura',1000,2,2))

