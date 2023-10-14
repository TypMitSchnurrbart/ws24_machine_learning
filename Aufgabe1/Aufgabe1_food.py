"""
    Test module
"""

#===== IMPORTS =======================================
import numpy
import pandas as pd
import json
import matplotlib.pyplot as plt

#===== FUNCTIONS =====================================
def hka():

    return


#===== MAIN ==========================================
if __name__ == "__main__":

    # Part One
    with open("foods.json", "r") as food_file:

        food_data = json.load(food_file)

    ## Create Dataframe with Indexs
    data = {
        "food_description" : [],
        "food_group" : [],
        "id" : [],
        "manufacturer" : []
    }

    for food in food_data:
        data["food_description"].append(food["description"])
        data["food_group"].append(food["group"])
        data["id"].append(food["id"])
        data["manufacturer"].append(food["manufacturer"])

    info = pd.DataFrame(data, columns=["food_description", "food_group", "id", "manufacturer"])


    ## Create Array of dataframes with nutrient list
    food_nutrients = []

    for food in food_data:

        nu_cache = {
        "description" : [],
        "group" : [],
        "units" : [],
        "value" : []
        }

        for nutrient in food["nutrients"]:
            nu_cache["description"].append(nutrient["description"])
            nu_cache["group"].append(nutrient["group"])
            nu_cache["units"].append(nutrient["units"])
            nu_cache["value"].append(nutrient["value"])

        element = pd.DataFrame(nu_cache, columns=["description", "group", "units", "value"])
        element["id"] = food["id"]
        food_nutrients.append(element)


    ## Concat the both
    nutrients = pd.concat(food_nutrients)
    nutrients.drop_duplicates()

    ## Join the two
    full_data_set = pd.merge(nutrients, info, on="id", how="outer")

    ## Read Zink data
    zink_data = full_data_set.loc[full_data_set["description"] == "Zinc, Zn"]

    ## Make statistics with the zinc data
    print(f"Descriptic Stats:")
    print(zink_data["value"].describe())
    print("Every value in mg\n\n")

    ## Search for Edamer
    edam = zink_data.loc[zink_data["food_description"] == "Cheese, edam"]
    print(f"\nEdammer Stats:\n{edam.drop_duplicates()}\n\n")

    ## Search for max value
    max_index = zink_data["value"].idxmax()
    max_entry = zink_data.loc[max_index]

    print("\n\nFood with max Zinc value:")
    print(max_entry)


    ## Create Hist
    zink_data["value"].hist(bins=len(zink_data["value"]))
    plt.xlabel('Zinc value [mg]')
    plt.ylabel('Frequency')
    plt.title('Histogram of zinc value in food')
    plt.show()


