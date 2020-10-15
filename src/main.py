from flask import Flask
from flask import render_template
from flask import request

import os

import pandas as pd
import numpy as np

import h2o
from h2o.estimators import H2ORandomForestEstimator

from geopy.geocoders import Nominatim

import re

    
# características por defecto.
BARRIO = "Malasaña-Universidad, Centro(Madrid)"
NECESITA_REFORMA = 1
NUEVA_CONSTRUCCION = 0
M2 = 90
HABITACIONES = 2
BANOS = 2
PLANTA = 2
EXTERIOR = 1
ASCENSOR = 1
PARKING = 1
TRASTERO = 1
TERRAZA = 1
BALCON = 1      
PISCINA = 1 


# modelo H2O
h2o.init()
rf_h2o = H2ORandomForestEstimator()



# flask app
app=Flask(__name__)



# antes del primer request...
@app.before_first_request
def startup():
    global rf_h2o
    
    data = h2o.import_file("../outputs/3_houses_no_typologies_no_outliers.csv")
    
    X = data.col_names[1:-1]
    y = data.col_names[-1]

    train, test =data.split_frame([0.8], seed=1234)
    rf_h2o.train(X, y, training_frame=train)
    

    
# main app
@app.route("/", methods=["POST", "GET"])
def main():
    
    if request.method=="POST":
        s_neighborhood_id=request.form["s_neighborhood_id"]
        s_m2=request.form["s_m2"]
        s_n_rooms=request.form["s_n_rooms"]
        s_n_bathrooms=request.form["s_n_bathrooms"]
        s_floor=request.form["s_floor"]

        
        # se obtienen las coordenadas latitud y longitud del barrio introducido.
        geolocator = Nominatim(user_agent="http")

        locate = geolocator.geocode(s_neighborhood_id, timeout=7)
        s_latitude = locate.latitude
        s_longitude = locate.longitude
        
        
        # planta
        floor_dict = {"Sótano": -2,
                      "Semisótano": -1,
                      "Bajo": 0,
                      "Entreplanta": 0.5,
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

        floor = float(floor_dict[s_floor])
        
        
        # se reasigna para la prediccion
        m2 = int(s_m2)
        n_rooms = int(s_n_rooms)
        n_bathrooms = int(s_n_bathrooms)
        latitude = int(s_latitude)
        longitude = int(s_longitude)
        
        
        # Se asignan características por defecto para features que no influyen en el precio.
        is_renewal_needed = NECESITA_REFORMA
        is_new_development = NUEVA_CONSTRUCCION
        is_exterior = EXTERIOR
        has_lift = ASCENSOR
        has_parking = PARKING
        has_storage_room = TRASTERO
        has_terrace = TERRAZA
        has_balcony = BALCON
        has_pool = PISCINA
                 
        
        # piso
        
        vivienda = {'is_renewal_needed': is_renewal_needed,
                'is_new_development': is_new_development,
                'm2': m2,
                'n_rooms': n_rooms,
                'n_bathrooms': n_bathrooms,
                'floor': floor,
                'is_exterior': is_exterior,
                'has_lift': has_lift,
                'has_parking': has_parking,
                'has_storage_room': has_storage_room,
                'has_terrace': has_terrace, 
                'has_balcony': has_balcony,
                'has_pool': has_pool,
                "latitude": latitude,
                "longitude": longitude
               }
        
        piso = pd.DataFrame.from_dict(vivienda, orient='index').T
        piso.to_csv("../outputs/new_piso.csv")
        piso = h2o.import_file("../outputs/new_piso.csv")
        y_pred = rf_h2o.predict(piso)
        a = y_pred.as_data_frame()
        b = str(a["predict"])
        c = re.findall(r"\d+",b)
        precio = c[1]
        
        # prediccion
        return f"EL PRECIO ESTIMADO DE SU VIVIENDA ES: {precio}"

             
        
        return render_template("tasador_virtual.html",
                               model_results='',
                               s_neighborhood_id = s_neighborhood_id,
                               s_m2 = s_m2,
                               s_n_rooms = s_n_rooms,
                               s_n_bathrooms = s_n_bathrooms,
                               s_floor = s_floor
                              )
    
    else:
        # parametros por defecto
        return render_template("tasador_virtual.html",
                               model_results = '',
                               s_neighborhood_id = BARRIO,
                               s_m2 = M2,
                               s_n_rooms = HABITACIONES,
                               s_n_bathrooms = BANOS,
                               s_floor = PLANTA
                              )
                               

# solo en local
if __name__=='__main__':
    app.run(debug=False)