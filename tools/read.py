import csv
import os

def read_csv(path):
    with open(path, 'r') as f:
        reader = csv.reader(f)
        return list(reader)

root_path = '../data/CorrData/DST_part3d/csv/'

for file in os.listdir(root_path):
    if '.csv' not in file:
        continue
    print('file: ', file)
    csv_path = os.path.join(root_path, file)
    csv_data = read_csv(csv_path)
    print('len csv_data: ', len(csv_data))
    i = 0
    for row in csv_data:
        print('row: ', row)
        i += 1
        if i > 10:
            break

