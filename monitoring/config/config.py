from typing import Dict, Text

DATABASE_URI: Text = "postgresql://admin:admin@localhost:5432/monitoring_db"
DATA_COLUMNS: Dict = {
    "target_col": "Price",
    "prediction_col": "predictions",
    "num_features": [
        "Mileage",
        "Power",
        "Vehicle_brand_Jaguar",
        "Vehicle_brand_Kia",
        "Vehicle_brand_Lamborghini",
        "Vehicle_brand_Mercedes-Benz",
        "Vehicle_brand_Mazda",
        "Vehicle_brand_MINI",
        "Vehicle_brand_Skoda",
        "Vehicle_brand_Volvo",
        "Vehicle_brand_Inny",
        "Year_of_production_2020",
        "Year_of_production_2017",
        "Year_of_production_2014",
        "Year_of_production_2013",
        "Year_of_production_2000",
        "Year_of_production_2009",
        "Year_of_production_2006",
        "Year_of_production_2022",
        "Year_of_production_1999",
        "Year_of_production_1998",
        "Year_of_production_1997",
        "Fuel_type_Benzyna",
        "Fuel_type_Benzyna+LPG",
        "Fuel_type_Diesel",
        "Fuel_type_Hybryda",
        "Gearbox_Manualna",
        "Body_type_Kabriolet",
        "Body_type_Kompakt",
        "Body_type_Kombi",
        "Number_of_doors_5",
        "Number_of_doors_4",
        "Number_of_doors_2",
        "price_per_mileage",
        "power_to_price_ratio",
    ],
}
DATA_COLUMNS["columns"] = (
    DATA_COLUMNS["num_features"] +
    [DATA_COLUMNS["target_col"], DATA_COLUMNS["prediction_col"]]
)
