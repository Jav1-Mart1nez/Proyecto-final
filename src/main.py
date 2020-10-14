from flask import Flask
from flask import render_template
from flask import request
from flask import Markup

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import io
import os
import base64

import numpy as np
from numpy import genfromtxt

import h2o
from h2o.estimators import H2OXGBoostEstimator

from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

 

# necesario en pythonanywhere
PATH=os.path.dirname(os.path.abspath(__file__))



# características por defecto.
BARRIO = "Malasaña-Universidad, Centro(Madrid)"
NECESITA_REFORMA = False
NUEVA_CONSTRUCCIÓN = False
m2 = 90
HABITACIONES = 2
BAÑOS = 2
PLANTA = 2
EXTERIOR = True
ASCENSOR = True
PARKING = True
TRASTERO = True
TERRAZA = True
BALCÓN = True      
PISCINA = True 


# modelo H2O
xgbm_h2o = H2OXGBoostEstimator()



# flask app
app=Flask(__name__)



# antes del primer request...
@app.before_first_request
def startup():
    global xgbm_h2o
    
    #data=genfromtxt(PATH+'/data/titanic.csv', delimiter=',') # fuera de jupyter
    data = h2o.import_file("..\outputs\h2o_no_acotados.csv")
    
    X = data.col_names[1:-1]
    y = data.col_names[-1]

    train, test =data.split_frame([0.8], seed=1234)
    xgbm_h2o.train(X, y, training_frame=train)
    

    
# main app
@app.route("/", methods = ["POST", "GET"])
def main():
    
    if request.method == "POST":
        s_neighborhood_id = request.form["s_neighborhood_id"]
        s_sq_mt_built = request.form["s_sq_mt_built"]
        s_n_rooms = request.form["s_n_rooms"]
        s_n_bathrooms = request.form["s_n_bathrooms"]
        s_floor = request.form["s_floor"]

        
        # se obtienen las coordenadas latitud y longitud del barrio introducido.
        geolocator = Nominatim(user_agent="http")

        locate = geolocator.geocode(s_neighborhood_id, timeout=7)
        s_latitude = locate.latitude
        s_longitude = locate.longitude
        
        
        # planta
        floor_dict = {"Sótano": -3,
                   "Semisótano": -2,
                   "Entreplanta": -1,
                   "Bajo": 0,
                   "1ª": 1,
                   "2ª": 2,
                   "3ª": 3,
                   "4ª": 4,
                   "5ª": 5,
                   "6ª": 6,
                   "7ª": 7,
                   "8ª": 8,
                   "9ª": 9 
                  }

        floor = floor_dict[s_floor]
        
        
        # se reasigna para la prediccion
        sq_mt_built = int(s_sq_mt_built)
        n_rooms = int(s_n_rooms)
        n_bathrooms = int(s_n_bathrooms)
        floor = int(s_floor)
        latitude = int(s_latitude)
        longitude = int(s_longitude)
        
        
        # Se asignan características por defecto para features que no influyen en el precio.
        renewal_needed = NECESITA_REFORMA
        new_development = NUEVA_CONSTRUCCIÓN
        exterior = EXTERIOR
        has_lift = ASCENSOR
        has_parking = PARKING
        has_storage_room = TRASTERO
        has_terrace = TERRAZA
        has_balcony = BALCÓN
        has_pool = PISCINA
                 
        
        # pasajero
        piso=[[is_renewal_needed, is_new_development, sq_mt_built, n_rooms,
               n_bathrooms, floor, is_exterior, has_lift, has_parking,
               has_storage_room, has_terrace, has_balcony, has_pool, latitude, longitude]]
        
        
        # prediccion
        xgbm_h2o.predict(piso)
             
        
        return render_template("..html/tasador_virtual.html",
                               model_results='',
                               s_neighborhood_id = s_neighborhood_id,
                               s_sq_mt_built = s_sq_mt_built,
                               s_n_rooms = s_n_rooms,
                               s_n_bathrooms = s_n_bathrooms,
                               s_floor = s_floor
                              )
    
    else:
        # parametros por defecto
        return render_template("..html/tasador_virtual.html",
                               model_results = '',
                               s_neighborhood_id = BARRIO,
                               s_sq_mt_built = m2,
                               s_n_rooms = HABITACIONES,
                               s_n_bathrooms = BAÑOS,
                               s_floor = PLANTA
                              )
                               

# solo en local
if __name__=='__main__':
    app.run(debug=False)