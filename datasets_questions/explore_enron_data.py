#!/usr/bin/python3

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""
import joblib

enron_data = joblib.load(open(
    "D:\\ud120-projects\\final_project\\final_project_dataset_unix.pkl", "rb"))

# Q. How many data points(people) are in the dataset?
print(len(enron_data.keys()))

# Q. For each person, how many features are available?
# Method 1
print(len(enron_data["SKILLING JEFFREY K"]))
# Method 2
print(len(list(enron_data.items())[0][1]))

# Q. How many POIs are there?
print(sum((enron_data[user]["poi"] == 1 for user in enron_data)))

# Q. print poi_names.read()
poi_names = open("D:\\ud120-projects\\final_project\\poi_names.txt", "r")
file_read = poi_names.readlines()
print(len(file_read[2:]))
poi_names.close()

# Q. What is the total value of the stock belonging to James Prentice?
print(enron_data["PRENTICE JAMES"]["total_stock_value"])

# Q. How many email messages do we have from Wesley Colwell to persons of interest?
print(enron_data["COLWELL WESLEY"]["from_this_person_to_poi"])

# Q. What’s the value of stock options exercised by Jeffrey Skilling?
print(enron_data["SKILLING JEFFREY K"]["exercised_stock_options"])

print("**** Total Payments *****: ")
print(enron_data["SKILLING JEFFREY K"]["total_payments"])
print(enron_data["FASTOW ANDREW S"]["total_payments"])
print(enron_data["LAY KENNETH L"]["total_payments"])

# Q. Salary & Email avl:
count_s, count_e = 0, 0
# enron_list = list(enron_data.values())

for key, value in enron_data.items():
    salary = value["salary"]
    if salary != "NaN":  # "!=" compares value, "is not" compares memory address
        count_s += 1

    email = value["email_address"]
    if email != "NaN":
        count_e += 1

print(f"Salaries: {count_s}, Emails: {count_e}")

# Q. Percentage of people with NaN in total_payments
count_tp = 0
for key, value in enron_data.items():
    total_payments = value["total_payments"]
    if total_payments == "NaN":
        count_tp += 1
# tp_percent = count_tp * 100 // len(enron_data.keys())
print(f"NaN tp: {count_tp * 100 // len(enron_data.keys())}%")

# Q. % of POI woth NaN tp
count_tp_poi = 0
for key, value in enron_data.items():
    isPoi, poi_tp = value["poi"], value["total_payments"]
    if isPoi and poi_tp == "NaN":
        count_tp_poi += 1
print(f"NaN tp POI: {count_tp_poi * 100 // len(enron_data.keys())}%")

print(list(enron_data.values())[0])