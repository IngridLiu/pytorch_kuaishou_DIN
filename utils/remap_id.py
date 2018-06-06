import random
import pickle
import pandas as pd

from utils.cfg import data_root
from utils.data_os import pickle_load, pickle_dump

# load the data of load_index
def load_df_data(pkl_path, load_index):
    df = pickle_load(pkl_path)
    df = df[load_index]
    return df

# build the map of key and number
def build_map(df, col_name):
    key = sorted(df[col_name].unique().tolist())
    m = dict(zip(key, range(len(key))))
    return m, key

# change the key of index to number
def exchange_map(remap_df, remap_index, col_map):
    remap_df[remap_index] = remap_df[remap_index].map(lambda x: col_map[x])
    return remap_df

def remap_interaction_data(df, remap_indics, user_id_map, photo_id_map):
    # handle photo feature index
    df = exchange_map(df, remap_indics[0], user_id_map)
    df = exchange_map(df, remap_indics[1], photo_id_map)
    # resort interaction data
    df = df.sort_values(['user_id', "time"])
    df = df.reset_index(drop=True)
    return df


random.seed(1234)

# data root
train_root = data_root + "train/"
test_root = data_root + "test/"


# load train interaction data
train_interaction_index = ["user_id", "photo_id", "click", "time"]
train_interaction_pkl = train_root + "train_interaction.pkl"
train_interaction_df = load_df_data(train_interaction_pkl, train_interaction_index)

# load test interaction data
test_interaction_index = ["user_id", "photo_id", "time"]
test_interaction_pkl = test_root + "test_interaction.pkl"
test_interaction_df = load_df_data(test_interaction_pkl, test_interaction_index)

interaction_df = pd.merge(train_interaction_df, test_interaction_df, on = ["user_id", "photo_id"], how = "outer")

# get user_id map and photo_id_map
user_id_map, user_id_key = build_map(interaction_df, "user_id")
photo_id_map, photo_id_key = build_map(interaction_df, "photo_id")

# some count number
user_count, photo_count, interaction_count = len(user_id_map), len(photo_id_map), interaction_df.shape[0]

# remap index
remap_indics =["user_id", "photo_id"]

# photo feature index
photo_feature_index =["photo_id", "visual_feature","cover_words",
                      "face_ratio", "gender", "age", "face_score"]

#load train photo_feature data
train_photo_feature_pkl = train_root + "photos_feature.pkl"
train_photo_feature_df = load_df_data(train_photo_feature_pkl, photo_feature_index)
# remap train feature df data
train_photo_feature_df = exchange_map(train_photo_feature_df, remap_indics[1], photo_id_map)
# sort photo feature data
train_photo_feature_df = train_photo_feature_df.sort_values('photo_id')
train_photo_feature_df = train_photo_feature_df.reset_index(drop=True)

# remap train interaction data
train_interaction_df = remap_interaction_data(train_interaction_df, remap_indics, user_id_map, photo_id_map)
# save train data
train_remap_save_path = train_root + "remap.pkl"
train_data = [train_interaction_df, train_photo_feature_df, (user_count, photo_count, interaction_count), (user_id_key, photo_id_key)]
pickle_dump(train_data, train_remap_save_path)



#load test photo_feature data
test_photo_feature_pkl = test_root + "photos_feature.pkl"
test_photo_feature_df = load_df_data(test_photo_feature_pkl, photo_feature_index)
test_photo_feature_df = exchange_map(test_photo_feature_df, remap_indics[1], photo_id_map)
# resort test photo feature df
test_photo_feature_df = test_photo_feature_df.sort_values('photo_id')
test_photo_feature_df = test_photo_feature_df.reset_index(drop=True)

# remap test interaction data
test_interaction_df = remap_interaction_data(test_interaction_df, remap_indics, user_id_map, photo_id_map)

# save train data
test_remap_save_path = test_root + "remap.pkl"
test_data = [test_interaction_df, test_photo_feature_df]
pickle_dump(test_data, test_remap_save_path)



