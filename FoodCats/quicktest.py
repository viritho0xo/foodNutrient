def compare_files(file1, file2):
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        file1_lines = f1.readlines()
        file2_lines = f2.readlines()

    return file1_lines == file2_lines

#are_files_same = compare_files('FoodCats/cookedFoodWetCat.txt', 'FoodCats/rawFoodWetCat.txt')

# print("Are the files the same?", are_files_same)
wetCat = open("FoodCats/WetCat.txt", "r").readlines()
wetCat = [int(x.strip()) for x in wetCat]
dryCat = open("FoodCats/DryCat.txt", "r").readlines()
dryCat = [int(x.strip()) for x in dryCat]

# find unique values in a list
unique = []
for val in dryCat:
    if val not in unique:
        unique.append(val)
print(unique)
