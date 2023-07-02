import requests

car = {
    "Mileage": 1.0,
    "Power": 1.0,
    "Vehicle_brand_Jaguar": 1.0,
    "Vehicle_brand_Kia": 1.0,
    "Vehicle_brand_Lamborghini": 1.0,
    "Vehicle_brand_Mercedes-Benz": 1.0,
    "Vehicle_brand_Mazda": 1.0,
    "Vehicle_brand_MINI": 1.0,
    "Vehicle_brand_Skoda": 1.0,
    "Vehicle_brand_Volvo": 1.0,
    "Vehicle_brand_Inny": 1.0,
    "Year_of_production_2020": 1.0,
    "Year_of_production_2017": 1.0,
    "Year_of_production_2014": 1.0,
    "Year_of_production_2013": 1.0,
    "Year_of_production_2000": 1.0,
    "Year_of_production_2009": 1.0,
    "Year_of_production_2006": 1.0,
    "Year_of_production_2022": 1.0,
    "Year_of_production_1999": 1.0,
    "Year_of_production_1998": 1.0,
    "Year_of_production_1997": 1.0,
    "Fuel_type_Benzyna": 1.0,
    "Fuel_type_Benzyna+LPG": 1.0,
    "Fuel_type_Diesel": 1.0,
    "Fuel_type_Hybryda": 1.0,
    "Gearbox_Manualna": 1.0,
    "Body_type_Kabriolet": 1.0,
    "Body_type_Kompakt": 1.0,
    "Body_type_Kombi": 1.0,
    "Number_of_doors_5": 1.0,
    "Number_of_doors_4": 1.0,
    "Number_of_doors_2": 1.0,
    "price_per_mileage": 1.0,
    "power_to_price_ratio": 1.0,
}

url = "http://127.0.0.1:9696/predict"
try:
    response = requests.post(url, json=car)
    print(response)
    response.raise_for_status()
    result = response.json()
    print(result)
except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")
