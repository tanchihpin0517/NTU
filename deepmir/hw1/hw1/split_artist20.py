#################################################
# How to use this sample code?
# Way 1
# Step 1: download artist20 and unzip it to the folder you want
# Step 2: put this code into the same folder
# Step 3: run this code and train.txt & validation.txt will in 'data_path' folder
#
# Way 2
# Step 1: download artist20 and unzip it to the folder you want
# Step 2: set variable data_path to your artist20 path
# Step 3: set variable target_path as you like
# Step 3: run this code and train.txt & validation.txt will in target_path folder
#################################################

import os

########## adjust here ###############
data_path = './data/artist20/mp3s-32k'       # set your artist20 path
target_path = './data/artist20'             # you can change the output path of .txt file if you want
######################################

if not os.path.exists(target_path):
    os.mkdir(target_path)

def create_singer_data(albums, target_txt, singer, data_path):
    for album in albums:
        song_root = os.path.join(data_path, singer, album)
        songs = os.listdir(song_root)
        write_list = []
        for song in songs:
            file_title, file_type = os.path.splitext(song)
            if file_type in ['.wav', '.mp3']:
                file_path = os.path.abspath(os.path.join(song_root, song))
                write_list.append([file_path, singer, file_title])

        with open(target_txt, 'a', encoding="utf-8") as f:
            for L in write_list:
                f.writelines(f'{L[0]}\t{L[1]}\t{L[2]}\n')

def remove_exist_file(target_txt):
    with open(target_txt, 'w', encoding="utf-8") as f:
        f.writelines('')
        
remove_exist_file(os.path.join(target_path, 'train.tsv'))
remove_exist_file(os.path.join(target_path, 'validation.tsv'))

singers = os.listdir(data_path)
for singer in singers:
    train = []
    validation = []
    albums = os.listdir(os.path.join(data_path, singer))   
    valid_size = 1
    train_size = len(albums) - valid_size
    count = 0
    albums.sort()
    for album in albums:
        if count < train_size:
            train.append(album)
        else:
            validation.append(album)

        count += 1
    create_singer_data(train, os.path.join(target_path, 'train.tsv'), singer, data_path)
    create_singer_data(validation, os.path.join(target_path, 'validation.tsv'), singer, data_path)   

print('Completed')
