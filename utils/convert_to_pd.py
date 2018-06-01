import os
import pickle
import numpy as np
import pandas as pd

from utils.cfg import data_root



def dir_to_df(dir_path = "", visual_index = []):
    i = 0
    df = {}
    for file_name in os.listdir(dir_path):
        file_path = dir_path + "/" +file_name
        file_content = np.load(file_path)
        file_content = np.reshape(file_content, -1)
        file_name_content = {visual_index[0]:file_name, visual_index[1]:file_content }
        df[i] = file_name_content
        i += 1
    df = pd.DataFrame.from_dict(df, orient="index")
    return df

def file_to_df(file_path = "", file_index = []):
    with open(file_path, 'r') as file:
        i = 0
        df = {}
        for line in file:
            j = 0
            line_dic = {}
            line = line.rstrip("\n")
            items = line.split("\t")
            for item in items:
                line_dic[file_index[j]] = item
                j += 1
            df[i] = line_dic
            i += 1
        df = pd.DataFrame.from_dict(df, orient='index')
        return df

def face_file_to_df(face_file_path = "", file_index = []):
    with open(face_file_path, 'r') as file:
        i = 0
        df ={}
        for line in file:
            j = 0
            line_dic = {}
            line = line.rstrip("\n")
            items = line.split("\t")
            items = list(items)
            for item in items:
                if j == 0:
                    line_dic[file_index[j]] = item
                    j += 1
                else:
                    item = eval(item)
                    for k in range(len(item[0])):
                        line_dic[file_index[j]] = item[0][k]
                        j += 1
            df[i] = line_dic
            i += 1
        return df

def save_photo_feature_df(data_paths="", indics=[], interaction_df={}, save_path=""):
    df = {}
    for i in range(3):
        if i == 0:
            df = dir_to_df(str(data_paths[i]), indics[i])
            df = df[df["photo_id"].isin(interaction_df["photo_id"].unique())]
            df = df.reset_index(drop=True)
        elif i ==1:
            text_df = file_to_df(str(data_paths[i]), indics[i])
            text_df = text_df[text_df["photo_id"].isin(interaction_df["photo_id"].unique())]
            text_df = text_df.reset_index(drop=True)
            df = pd.merge(df, text_df, on = "photo_id")
        else:
            face_df = face_file_to_df(str(data_paths[i]), indics[i])
            face_df = face_df[face_df["photo_id"].isin(interaction_df["photo_id"].unique())]
            face_df = face_df.reset_index(drop=True)
            df = pd.merge(pd, face_df, on = "photo_id")

    with open(save_path, 'wb') as file:
        pickle.dump(df, file, pickle.HIGHEST_PROTOCOL)


# save train data

train_root = data_root + "train/"

interaction_index = ["user_id", "photo_id", "click", "like", "follow", "time", "playing_time", "duration_time"]
visual_index = ["photo_id", "visual_feature"]
text_index = ["photo_id", "cover_words"]
face_index = ["photo_id", "face_ratio", "gender", "age", "face_score"]
photo_feature_indics = [visual_index, text_index, face_index]

# file path
visual_dir = train_root + "preliminary_visual_train/"
text_file_path = train_root + "train_text.txt"
face_file_path = train_root + "train_face.txt"
train_photos_feature_paths = [visual_dir, text_file_path, face_file_path]

# save train interaction data to train_interaction.pkl
train_interaction_file = train_root + "train_interaction.txt"
train_interaction_df = file_to_df(train_interaction_file, interaction_index)
train_interaction_df_file = train_root + "train_interaction.pkl"
with open(train_interaction_df_file, 'wb') as file:
    pickle.dump(train_interaction_df_file, file, pickle.HIGHEST_PROTOCOL)

train_save_path = train_root + "photos_feature.pkl"
save_photo_feature_df(data_paths = train_photos_feature_paths,
                      indics = photo_feature_indics,
                      interaction_df = train_interaction_df,
                      save_path=train_save_path)

# save test data

test_root = data_root + "test/"

test_interaction_index = ["user_id", "photo_id", "time", "duration_time"]

visual_dir = test_root + "preliminary_visual_train/"
text_file_path = test_root + "train_text.txt"
face_file_path = test_root + "train_face.txt"
test_photos_feature_paths = [visual_dir, text_file_path, face_file_path]

# save test interaction data to test_interaction.pkl
test_interaction_file = test_root + "test_interaction.txt"
test_interaction_df = file_to_df(test_interaction_file, test_interaction_index)
test_interaction_df_file = test_root + "test_interaction.pkl"
with open(test_interaction_df_file, 'wb') as file:
    pickle.dump(test_interaction_df_file, file, pickle.HIGHEST_PROTOCOL)

test_save_path = test_root + "photos_feature.pkl"
save_photo_feature_df(data_paths = test_photos_feature_paths,
                      indics = photo_feature_indics,
                      interaction_df = test_interaction_df,
                      save_path=test_save_path)
