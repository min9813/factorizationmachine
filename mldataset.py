import os
import pandas as pd
import torch
from torch.utils.data import Dataset

MLPATH = "/home/minteiko/developer/project/data/ml-20m"
DATAPATH = os.path.join(MLPATH, "ratings.csv")
SMALL_DATAPATH = os.path.join(MLPATH, "small_ratings.csv")


def make_small_data(data_path, save_path):
    ratings = pd.read_csv(data_path)
    use_userid = ratings.userId.value_counts()[:100]
    total_length = use_userid.sum()
    print("use data #:{} , ratio to original: {:.1f}%".format(
        total_length, total_length/float(ratings.shape[0])*100))
    use_userid = use_userid.index
    small_rating_data = ratings[ratings.userId.isin(use_userid)].reset_index().drop("index", axis=1)
    small_rating_data.to_csv(save_path, index=False)


class MLDataset(Dataset):

    def __init__(self, csv_path):
        self.dataset = pd.read_csv(csv_path)
        self.feature_num = 0
        self.field_num = 0
        self.preprocessing()
        self.dataset = torch.FloatTensor(self.dataset.values)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def preprocessing(self):
        unique_user_length = len(self.dataset.userId.unique())
        assert unique_user_length < 101
        user_id = sorted(self.dataset["userId"].unique())
        user_id_map_dict = {}
        for user in user_id:
            user_id_map_dict[user] = len(user_id_map_dict)
        movie_id = sorted(self.dataset["movieId"].unique())
        movie_id_map_dict = {}
        for movie in movie_id:
            movie_id_map_dict[movie] = len(
                movie_id_map_dict) + unique_user_length
        self.dataset["userId"] = self.dataset["userId"].map(user_id_map_dict)
        self.dataset["movieId"] = self.dataset["movieId"].map(movie_id_map_dict)
        self.dataset = self.dataset.sort_values(by="timestamp")
        self.dataset.drop(["timestamp"], axis=1,inplace=True)
        self.feature_num = self.dataset["movieId"].max()+1
        self.field_num = 2


if __name__ == "__main__":
    small_data = make_small_data(DATAPATH, SMALL_DATAPATH)
