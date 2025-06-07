import requests

url = "https://api.openweathermap.org/data/2.5/weather"
params = {
    "lat": -7.791594544974983,
    "lon": 110.3662733746616,
    "appid": "b3a7fc753e5585d726234e2088421c5b",
    "units": "metric"
}

try:
    response = requests.get(url, params=params)
    print(response.json())
except requests.exceptions.RequestException as e:
    print(e)
