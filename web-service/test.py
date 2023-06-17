import requests

ride = {
    "PULocationID": 10,
    "DOLocationID": 50,
    "trip_distance": 40
}

url = 'http://127.0.0.1:9696/predict'
try:
    response = requests.post(url, json=ride)
    # print(response)
    response.raise_for_status()  # Raise an exception if the request was not successful
    result = response.json()
    print(result)
except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")
