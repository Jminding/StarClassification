import pandas as pd
import numpy as np
import os
import shutil

df = pd.read_csv('star_data_3.csv')
sp = df["Luminosity Class"].to_list()

def move_file(source_directory, destination_directory, filename):
    # Create the destination directory if it doesn't exist
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)

    # Construct the full paths for the source and destination files
    source_path = os.path.join(source_directory, filename)
    destination_path = os.path.join(destination_directory, filename)

    try:
        # Move the file
        shutil.move(source_path, destination_path)
        print(f"File '{filename}' moved successfully from {source_directory} to {destination_directory}")
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found in {source_directory}")
    except Exception as e:
        print(f"An error occurred: {e}")

for i in list(df["Star Name"]):
    # shutil.move(f"Star Images/{i}_removedbg.jpg", f"Star Images/Removed Background/{i}_removedbg.jpg")
    # open(f"Star Images/{eval(df[df['Star Name'] == i]['Spectral Class'].values[0])[0]}/{i}.jpg", "w")
    # os.system(f'mv "Star Images/{i}.jpg" "Star Images/{eval(df[df["Star Name"] == i]["Spectral Class"].values[0])[0]}/{i}.jpg"')
    if str(df[df['Star Name'] == i]['Luminosity Class'].values[0]) == "nan":
        continue
    for j in ["", "_horizontalflipped", "_verticalflipped", "_originflipped", "_90", "_180", "_270", "_horizontalflipped_90", "_horizontalflipped_180", "_horizontalflipped_270", "_verticalflipped_90", "_verticalflipped_180", "_verticalflipped_270", "_originflipped_90", "_originflipped_180", "_originflipped_270"]:
        move_file(f"Star Images LC", f"Star Images LC/{df[df['Star Name'] == i]['Luminosity Class'].values[0]}", f"{i}{j}.jpg")
        # print(i)


# shutil.move()