import pickle
import json
import numpy as np
import os

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '.'))


with open(os.path.join(ROOT_DIR, "artifacts/columns.json"), "r") as f:
    __data__Columns = json.load(f)['data_columns']
    __locations = __data__Columns[3:]

with open(os.path.join(ROOT_DIR, "artifacts/banglore_home_prices_model.pickle"), 'rb') as f:
    __model = pickle.load(f)

def get_location_names():
    return __locations

def get_data_columns():
    return __data__Columns

def get_estimated_price(location,sqft,bhk,bath):
    try:
        loc_index = __data__Columns.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(__data__Columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index>0:
        x[loc_index] = 1
    return round(__model.predict([x])[0],2)

if __name__ == "__main__":
    print(get_location_names())
    print(get_estimated_price('1st Phase JP Nagar', 1000,3, 3))
    print(get_estimated_price('1st Phase JP Nagar', 1000, 2, 2))
    print(get_estimated_price('subramanyapura', 1000,2,2))
    print(get_estimated_price('vijayanagar',1000,2,2))






























