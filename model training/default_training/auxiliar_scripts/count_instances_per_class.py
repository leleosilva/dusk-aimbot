"""
Script that searches for YOLO label files (.txt) and, for each file,
counts the instances of each class labeled.
It then prints the total instances for each class.

Useful for keeping the dataset balanced (ensuring a similar number of samples for each class).
"""

from glob import glob
import os.path

# Make an empty list to count for each class
count = [0, 0, 0, 0, 0]
PATH = "../image_data/labels"

# For each text file in the label directory
for filepath in glob(f"{PATH}/*.txt"):
    with open(filepath, 'r') as file:
        rows = file.read().split("\n")
        for row in rows:
            if len(row) > 0:
                try: # The first element in each line is the index which corresponds to the class number
                    count[int(row[0])] += 1
                except:
                    print(row) # Prints the class name for each class

print(count)
