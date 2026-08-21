"""Schema/constants shared across the app: raw input columns + label lists."""

RAW_NUMERIC_COLUMNS = [
    "Electricity:Facility [kW](Hourly)",
    "Fans:Electricity [kW](Hourly)",
    "Cooling:Electricity [kW](Hourly)",
    "Heating:Electricity [kW](Hourly)",
    "InteriorLights:Electricity [kW](Hourly)",
    "InteriorEquipment:Electricity [kW](Hourly)",
    "Gas:Facility [kW](Hourly)",
    "Heating:Gas [kW](Hourly)",
    "InteriorEquipment:Gas [kW](Hourly)",
    "Water Heater:WaterSystems:Gas [kW](Hourly)",
]


BUILDING_CLASSES = [
    "FullServiceRestaurant", "Hospital", "LargeHotel", "LargeOffice",
    "MediumOffice", "MidriseApartment", "OutPatient", "PrimarySchool",
    "QuickServiceRestaurant", "SecondarySchool", "SmallHotel", "SmallOffice",
    "Stand-aloneRetail", "StripMall", "SuperMarket", "Warehouse",
]


THEFT_LABELS = ["Normal", "Theft1", "Theft2", "Theft3", "Theft4", "Theft5", "Theft6"]

REQUIRED_BATCH_COLUMNS = ["row_id"] + RAW_NUMERIC_COLUMNS + ["Class"]
