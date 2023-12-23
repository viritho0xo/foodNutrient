import requests
import json
import pandas as pd

# API endpoint for SR-legacy database
url = " https://api.nal.usda.gov/fdc/v1/food/"

# API key for accessing the database
api_key = "SCt5LaywcTc3wc8zPamq1ODlcz1g29EoEduQW7TR"

nutrients = 'Water Protein Total lipid (fat) Carbohydrate, by difference Calcium, Ca Iron, Fe Magnesium, Mg Phosphorus, P Potassium, K Sodium, Na Zinc, Zn Copper, Cu Manganese, Mn Selenium, Se Vitamin C, total ascorbic acid Thiamin Riboflavin Niacin Pantothenic acid Vitamin B-6 Folate, total Vitamin B-12 Vitamin A, RAE Cholesterol Fatty acids, total saturated Fatty acids, total monounsaturated Fatty acids, total polyunsaturated'
def fetch_food_details(fdcic):
    # Send the API request
    # response1 = requests.get(url + str(
    #     fdcic) + "?format=full&nutrients=203,204,205,255,301,303,304,305,306,307,309,312,315,317,320,401,404,405,406,"
    #              "410,415,417,418,601&api_key=" + api_key)
    response2 = requests.get(url + str(fdcic) + "?format=full&nutrients=606,645,646&api_key=" + api_key)
    # Check if the request was successful
    if response2.status_code == 200:
        rawAmount = [0]*27
        # nameSequence = ""
        # data1 = response1.json()
        data2 = response2.json()
        # nutrients1 = data1['foodNutrients']
        # nutrients2 = data2['foodNutrients']
        foodCat = data2['foodCategory']['id']
        # for nutrient in nutrients1:
        #     match nutrient['nutrient']['name']:
        #         case "Water":
        #             rawAmount[0] = nutrient['amount']
        #         case "Protein":
        #             rawAmount[1] = nutrient['amount']
        #         case "Total lipid (fat)":
        #             rawAmount[2] = nutrient['amount']
        #         case "Carbohydrate, by difference":
        #             rawAmount[3] = nutrient['amount']
        #         case "Calcium, Ca":
        #             rawAmount[4] = nutrient['amount']
        #         case "Iron, Fe":
        #             rawAmount[5] = nutrient['amount']
        #         case "Magnesium, Mg":
        #             rawAmount[6] = nutrient['amount']
        #         case "Phosphorus, P":
        #             rawAmount[7] = nutrient['amount']
        #         case "Potassium, K":
        #             rawAmount[8] = nutrient['amount']
        #         case "Sodium, Na":
        #             rawAmount[9] = nutrient['amount']
        #         case "Zinc, Zn":
        #             rawAmount[10] = nutrient['amount']
        #         case "Copper, Cu":
        #             rawAmount[11] = nutrient['amount']
        #         case "Manganese, Mn":
        #             rawAmount[12] = nutrient['amount']
        #         case "Selenium, Se":
        #             rawAmount[13] = nutrient['amount']
        #         case "Vitamin C, total ascorbic acid":
        #             rawAmount[14] = nutrient['amount']
        #         case "Thiamin":
        #             rawAmount[15] = nutrient['amount']
        #         case "Riboflavin":
        #             rawAmount[16] = nutrient['amount']
        #         case "Niacin":
        #             rawAmount[17] = nutrient['amount']
        #         case "Pantothenic acid":
        #             rawAmount[18] = nutrient['amount']
        #         case "Vitamin B-6":
        #             rawAmount[19] = nutrient['amount']
        #         case "Folate, total":
        #             rawAmount[20] = nutrient['amount']
        #         case "Vitamin B-12":
        #             rawAmount[21] = nutrient['amount']
        #         case "Vitamin A, RAE":
        #             rawAmount[22] = nutrient['amount']
        #         case "Cholesterol":
        #             rawAmount[23] = nutrient['amount']
        
        # for nutrient in nutrients2:
        #     match nutrient['nutrient']['name']:
        #         case "Fatty acids, total saturated":
        #             rawAmount[24] = nutrient['amount']
        #         case "Fatty acids, total monounsaturated":
        #             rawAmount[25] = nutrient['amount']
        #         case "Fatty acids, total polyunsaturated":
        #             rawAmount[26] = nutrient['amount']
        
        return foodCat
    else:
        # If the request was not successful, raise an exception
        response2.raise_for_status()
# write to file
# print(json.dumps(out, indent=4, sort_keys=True), file=open("food_details.txt", "a"))

# with open('food_details_out.txt', 'w', encoding="utf-8") as f:
#     out = fetch_food_details("169961")
#     f.writelines(out["fdcId"] + "\n" + out["nameSequence"] + "\n" + out["rawAmount"] + "\n")

rawFoodDry = open("rawListDry.txt", "r", encoding="utf-8").read().split()
rawFoodWet = open("rawListWet.txt", "r", encoding="utf-8").read().split()
cookedFoodDry = open("cookedListDry.txt", "r", encoding="utf-8").read().split()
cookedFoodWet = open("cookedListWet.txt", "r", encoding="utf-8").read().split()

for food in cookedFoodWet:
    out = str(fetch_food_details(food))
    with open('FoodCats/cookedFoodWetCat.txt', 'a', encoding="utf-8") as f:
        f.writelines(out + "\n")

# for food in rawFoodWet:
#     out = str(fetch_food_details(food))
#     with open('FoodCats/rawFoodWeCat.txt', 'a', encoding="utf-8") as f:
#         f.writelines(out + "\n")

# for food in cookedFoodDry:
#     out = fetch_food_details(food)
#     with open('FoodNutrients/cookedFoodDryNutri.txt', 'a', encoding="utf-8") as f:
#         f.writelines(out + "\n")
# for food in cookedFoodWet:
#     out = fetch_food_details(food)
#     with open('FoodNutrients/cookedFoodWetNutri.txt', 'a', encoding="utf-8") as f:
#         f.writelines(out + "\n")



# countSpace = open("FoodNutrients/rawFoodDryNutri.txt", "r", encoding="utf-8").readlines()
#
#
# def check_space(string):
#     # counter
#     count = 0
#
#     # loop for search each index
#     for i in range(0, len(string)):
#
#         # Check each char
#         # is blank or not
#         if string[i] == " ":
#             count += 1
#
#     return count
#
# for food in countSpace:
#     print(check_space(food))
