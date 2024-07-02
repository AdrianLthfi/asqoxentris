import datetime

import joblib
import numpy as np
import requests
import xarray as xr
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

model = joblib.load('log_reg_model.pkl')
vv_ds_processed = xr.open_dataset('vv_data_processed.nc')
vh_ds_processed = xr.open_dataset('vh_data_processed.nc')

latitudes = np.linspace(vv_ds_processed.coords['y'].values[0], vv_ds_processed.coords['y'].values[-1], vv_ds_processed.sizes['y'])
longitudes = np.linspace(vv_ds_processed.coords['x'].values[0], vv_ds_processed.coords['x'].values[-1], vv_ds_processed.sizes['x'])
features = (vv_ds_processed['VV'] / (vh_ds_processed['VH'] + 1e-6)).values

user_location = {'latitude': None, 'longitude': None}

def find_nearest_index(lat, lon, latitudes, longitudes):
    distances = np.sqrt((latitudes - lat)**2 + (longitudes - lon)**2)
    return np.argmin(distances)

def get_earthquake_data(days):
    endtime = datetime.datetime.utcnow().isoformat()
    starttime = (datetime.datetime.utcnow() - datetime.timedelta(days=days)).isoformat()
    url = f'https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&starttime={starttime}&endtime={endtime}&minlatitude=-11.0&maxlatitude=6.0&minlongitude=95.0&maxlongitude=141.0'
    response = requests.get(url)
    data = response.json()
    return data['features']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/map')
def map_view():
    return render_template('map.html')

@app.route('/dashboard/')
def dashboard_view():
    return render_template('dashboard.html')

@app.route('/map_data', methods=['GET'])
def map_data():
    try:
        days = int(request.args.get('days', 30))
        earthquakes = get_earthquake_data(days)
        
        latest_time = max(eq['properties']['time'] for eq in earthquakes)
        earthquake_data = [{
            'coords': eq['geometry']['coordinates'],
            'magnitude': eq['properties']['mag'],
            'place': eq['properties']['place'],
            'time': datetime.datetime.fromtimestamp(eq['properties']['time'] / 1000).strftime('%Y-%m-%d %H:%M:%S'),
            'is_latest': eq['properties']['time'] == latest_time
        } for eq in earthquakes]

        return jsonify(earthquake_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['GET'])
def predict():
    latitude = request.args.get('latitude', type=float)
    longitude = request.args.get('longitude', type=float)
    magnitude = request.args.get('magnitude', type=float)
    depth = request.args.get('depth', type=float)
    fault_type = request.args.get('fault_type', type=str)
    
    if magnitude < 6.5 or depth > 30 or fault_type not in ['sesar naik', 'sesar turun']:
        risk_level = 0
    else:
        index = find_nearest_index(latitude, longitude, latitudes, longitudes)
        radar_feature_value = features.ravel()[index]
        input_features = np.array([[magnitude, depth, radar_feature_value]])
        risk_level = model.predict(input_features)[0]

    return jsonify({'tsunami_risk': int(risk_level)})

@app.route('/chatbot', methods=['POST'])
def chatbot():
    user_message = request.json.get('message').lower()
    response = "Maaf, saya tidak mengerti. Silakan coba pertanyaan lain."

    if "gempa" in user_message:
        try:
            earthquakes = get_earthquake_data(1)
            if earthquakes:
                latest_earthquake = earthquakes[0]
                place = latest_earthquake['properties']['place']
                magnitude = latest_earthquake['properties']['mag']
                time = datetime.datetime.fromtimestamp(latest_earthquake['properties']['time'] / 1000).strftime('%Y-%m-%d %H:%M:%S')
                response = f"Gempa terbaru terjadi di {place} dengan kekuatan {magnitude} pada {time}."
            else:
                response = "Data gempa terkini tidak tersedia saat ini, mohon coba lagi nanti."
        except Exception as e:
            response = f"Terjadi kesalahan saat mengambil data gempa: {str(e)}"

    elif "tsunami" in user_message:
        response = "Saya belum memiliki informasi terkini mengenai tsunami. Silakan coba lagi nanti."

    return jsonify({'response': response})

@app.route('/user_location', methods=['POST'])
def user_location():
    data = request.json
    user_location['latitude'] = data.get('latitude')
    user_location['longitude'] = data.get('longitude')
    return jsonify({'status': 'success', 'latitude': user_location['latitude'], 'longitude': user_location['longitude']})

@app.route('/dashboard_data', methods=['GET'])
def dashboard_data():
    try:
        days = int(request.args.get('days', 1))

        earthquakes = get_earthquake_data(days)
        earthquake_data = [{
            'place': eq['properties']['place'],
            'magnitude': eq['properties']['mag'],
            'depth': eq['geometry']['coordinates'][2]
        } for eq in earthquakes]

        return jsonify(earthquake_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
