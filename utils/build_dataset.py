import random
import pickle

from utils.cfg import data_root
from utils.data_os import pickle_load, pickle_dump


# 获取df[photo_id, click)中click值为1的photo_id,
def get_pos_photo_list(photo_df):
    pos_photo_id = []
    for index, row in photo_df.iterrows():
        if row["click"] == 1:
            pos_photo_id.append(row["photo_id"])
    return pos_photo_id

# data root
train_root = data_root + "train/"
test_root  = data_root + "test/"

# the path saved remap  data
train_remap_path = train_root + "train_remap.pkl"
test_remap_path = test_root + "test_remap.pkl"

# load train pkl data
train_data = pickle_load(train_remap_path)
train_interaction_df = train_data[0]
train_photo_feature_df = train_data[1]
user_count, photo_count, interaction_count = train_data[2]

# load test pkl data
test_data = pickle_load(test_remap_path)
test_interaction_df = test_data[0]
test_photo_feature_df = test_data[1]


train_set = []
val_set = []
test_set = []

# get train set and val set
for user_id, hist in train_interaction_df.groupby("user_id"):
    hist_photo_df = hist[["photo_id", "click"]]
    hist_pos_photo_list = get_pos_photo_list(hist_photo_df)

    def gen_neg():
        neg = hist_pos_photo_list[0]
        while neg in hist_pos_photo_list:
            neg = random.randint(0, photo_count - 1)
        return neg

    hist_neg_photo_list = [gen_neg() for i in range(len(hist_pos_photo_list))]

    for i in range(1, len(hist_pos_photo_list)):
        hist = hist_pos_photo_list[:i]
        if i != len(hist_pos_photo_list) - 1:
            train_set.append((user_id, hist, hist_pos_photo_list[i], 1))
            train_set.append((user_id, hist, hist_neg_photo_list[i], 0))
        else:
            label = (hist_pos_photo_list[i], hist_neg_photo_list[i])
            val_set.append((user_id, hist, hist_pos_photo_list[i], 1))
            val_set.append((user_id, hist, hist_neg_photo_list[i], 0))

random.shuffle(train_set)
random.shuffle(val_set)

# get test set
val_user_info = [(x[:2]) for x in val_set]
nodup_val_user_info = []
nodup_val_user_id = []
for user_info in val_user_info:
    if user_info[0] not in nodup_val_user_id:
        nodup_val_user_id.append(user_info[0])
        nodup_val_user_info.append(user_info)

for index,row in test_interaction_df.iterrows():
    user_id = row["user_id"]
    hist = []
    if user_id in nodup_val_user_id:
        hist = val_set[nodup_val_user_id.index(user_id)][1]
    test_set.append((user_id, hist))

random.shuffle(test_set)



# save train and val set
train_save_path = train_root + "train_dataset.pkl"
train_data_set = [train_set, val_set, train_photo_feature_df, (user_count, photo_count)]
pickle_dump(train_data_set, train_save_path)

# save test set
test_save_path = test_root + "test_dataset.pkl"
test_data_set = [test_set, test_photo_feature_df]
pickle_dump(test_data_set, test_save_path)