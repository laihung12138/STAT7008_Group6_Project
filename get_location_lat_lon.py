import pandas as pd
import requests

def get_lat_lon(address, api_key):
    url = f"https://maps.googleapis.com/maps/api/geocode/json?address={address}&key={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data['results']:
            location = data['results'][0]['geometry']['location']
            return location['lat'], location['lng']
    return None, None # Return None if the address is not found

def get_lat_lon_columns(address_column, api_key):
    latitudes = []
    longitudes = []

    for address in address_column:
        lat, lon = get_lat_lon(address, api_key)
        latitudes.append(lat)
        longitudes.append(lon)

    return latitudes, longitudes

if __name__ == "__main__":
    data = {'Address': ['The University of Hong Kong', 'CUHK', 'HKUST']}
    df = pd.DataFrame(data)

    api_key = 'AIzaSyAJpVqYsbzLgLS4dRRnqOgfS_3utQxs0wo'

    df_with_lat_lon = get_lat_lon_columns(df['Address'], api_key)
    print(df_with_lat_lon)