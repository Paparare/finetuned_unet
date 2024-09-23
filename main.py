import os
import csv

directory = 'pic/'

train_directory = os.path.join(directory, 'input')
test_directory = os.path.join(directory, 'output')

os.makedirs(train_directory, exist_ok=True)
os.makedirs(test_directory, exist_ok=True)

train_csv_path = os.path.join(directory, 'input.csv')
test_csv_path = os.path.join(directory, 'output.csv')


def extract_weft_warp(filename):
    parts = filename.split('经线')
    if len(parts) < 2:
        return None, None, None
    id_part = parts[0].strip()
    rest = parts[1].split('纬线')
    if len(rest) < 2:
        return None, None, None
    warp = rest[0].strip()
    weft = rest[1].split('-')[0].strip()
    return id_part, warp, weft


with open(train_csv_path, mode='w', newline='', encoding='utf-8') as train_file, \
        open(test_csv_path, mode='w', newline='', encoding='utf-8') as test_file:
    train_writer = csv.writer(train_file)
    test_writer = csv.writer(test_file)

    train_writer.writerow(['ID', 'Warp', 'Weft'])
    test_writer.writerow(['ID', 'Warp', 'Weft'])

    for filename in os.listdir(directory):
        if filename.endswith('.png') or filename.endswith('.jpg'):  # Adjust file types as necessary
            id_part, warp, weft = extract_weft_warp(filename)

            if id_part and warp and weft:
                if '-1' in filename:
                    new_name = f"{id_part}{os.path.splitext(filename)[1]}"
                    os.rename(os.path.join(directory, filename), os.path.join(test_directory, new_name))
                    test_writer.writerow([id_part, warp, weft])
                else:
                    new_name = f"{id_part}{os.path.splitext(filename)[1]}"
                    os.rename(os.path.join(directory, filename), os.path.join(train_directory, new_name))
                    train_writer.writerow([id_part, warp, weft])

train_csv_path, test_csv_path
