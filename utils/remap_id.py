import random
import pickle

from utils.cfg import data_root

def load_df_data(pkl_path, load_index):
    with open(pkl_path, 'rb') as file:
        train_df = pickle.load(file)
        train_df = train_df[load_index]
        return train_df

def build_map(df, col_name):
    key = sorted(df[col_name].unique().tolist())
    m = dict(zip(key, range(len(key))))
    return m, key

def exchange_map(remap_df, remap_index, col_map):
    remap_df[remap_index] = remap_df[remap_index].map(lambda x: col_map[x])
    return remap_df





random.seed(1234)

train_root = data_root + "train/"

train_interaction_index = ["user_id", "photo_id", "click", "time"]
train_interaction_pkl = train_root + "train_interaction.pkl"
train_interaction_df = load_df_data(train_interaction_pkl, train_interaction_index)

train_photo_feature_inndex =["photo_id", "visual_feature","cover_words"
                             "face_ratio", "gender", "age", "face_score"]
train_photo_feature_pkl = train_root + "photos_feature.pkl"
train_photo_feature_df = load_df_data(train_photo_feature_pkl, train_photo_feature_inndex)

# get user_id map and photo_id_map
user_id_map, user_id_key = build_map(train_interaction_df, "user_id")
photo_id_map, photo_id_key = build_map(train_interaction_df, "photo_id")

# remap train data
remap_indics =["user_id", "photo_id"]
train_interaction_df = exchange_map(train_interaction_df, remap_indics[0], user_id_map)
train_interaction_df = exchange_map(train_interaction_df, remap_indics[1], photo_id_map)
train_photo_feature_df = exchange_map(train_photo_feature_df, remap_indics[1], photo_id_map)

user_count, photo_count, interaction_count = len(user_id_map), len(photo_id_map), train_interaction_df.shape[0]


train_photo_feature_df = train_photo_feature_df.sort_values('user_id')
train_photo_feature_df = train_photo_feature_df.reset_index(drop=True)

train_interaction_df["user_id"] = train_interaction_df["user_id"].map(lambda x: user_id_map[x])
train_interaction_df = train_interaction_df.sort_values(['user_id', "time"])
train_interaction_df = train_interaction_df.reset_index(drop = True)

remap_save_path = train_root + "remap.pkl"
with open(remap_save_path, 'wb') as file:
    pickle.dump(train_photo_feature_df, file, pickle.HIGHEST_PROTOCOL)
    pickle.dump(train_interaction_df, file, pickle.HIGHEST_PROTOCOL)
    pickle.dump((user_count, photo_count, interaction_count), file, pickle.HIGHEST_PROTOCOL)
    pickle.dump((user_id_key, photo_id_key), file, pickle.HIGHEST_PROTOCOL)
