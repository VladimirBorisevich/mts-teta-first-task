import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import yaml


def read_yaml(path):
    dir_path = os.path.dirname(os.path.dirname(os.getcwd()))
    dir_path = os.path.join(dir_path, 'config')
    return yaml.safe_load(open(os.path.join(dir_path, path)))['project']


def read_data(path):
    return pd.read_csv(path)


def prep_date(data):
    data["date_time"] = data["date"] + " " + data["time"]
    data = data.drop(["date", "time"], axis=1)
    data['date_time'] = pd.to_datetime(data["date_time"])
    data["day_delta"] = (data["date_time"] - data.date_time.min()).dt.days
    # data = data.drop("date_time", axis=1)
    return data


def clean_data(data):
    data = data[(data.price > 0) & (data.area > 0) & (data.kitchen_area > 0)]
    data = data.drop_duplicates()
    data.reset_index(inplace=True, drop=True)
    return data


def drop_regions(data, amount_of_regions):
    regions_to_drop = data.groupby('region')['region'].count().sort_values().head(amount_of_regions).index.to_list()
    data = data[~data['region'].isin(regions_to_drop)]
    return data


def filter_by_region(data, amount_of_std, feature):
    def clean(reg):
        std_shift = data.loc[data.region == reg][feature].std() * amount_of_std
        feat_mean = data.loc[data.region == reg][feature].mean()
        lower_bound = feat_mean - std_shift
        upper_bound = feat_mean + std_shift
        rows_to_drop = np.where((data.loc[data.region == reg, feature] < lower_bound) | \
                                (data.loc[data.region == reg, feature] >= upper_bound))[0]
        indexes = data.loc[data.region == reg].iloc[rows_to_drop].index
        data.drop(indexes, axis=0, inplace=True)
        data.reset_index(inplace=True, drop=True)
        return data

    for region in tqdm(data.region.unique()):
        data = clean(region)
    return data


def convert_object_type(data):
    data["object_type"] = data["object_type"].map({1: 0, 11: 1}).astype(int)
    return data


def make_time_features(data):
    data["hour"] = data['date_time'].apply(lambda x: x.hour)
    data["year"] = data["date_time"].apply(lambda x: x.year)
    return data


def split_data(data, train_size):
    data = data.sort_values(by="date_time")
    train_df = data.iloc[:round(data.shape[0] * train_size)]
    test_df = data.iloc[round(data.shape[0] * train_size):]
    return train_df, test_df


def reduce_mem_usage(data, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = data.memory_usage().sum() / 1024 ** 2
    for col in data.columns:
        col_type = data[col].dtypes
        if col_type in numerics:
            c_min = data[col].min()
            c_max = data[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    data.loc[:, col] = data.loc[:, col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    data.loc[:, col] = data.loc[:, col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    data.loc[:, col] = data.loc[:, col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    data.loc[:, col] = data.loc[:, col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    data.loc[:, col] = data.loc[:, col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    data.loc[:, col] = data.loc[:, col].astype(np.float32)
                else:
                    data.loc[:, col] = data.loc[:, col].astype(np.float64)
    end_mem = data.memory_usage().sum() / 1024 ** 2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem,
                                                                              100 * (start_mem - end_mem) / start_mem))
    return data


def save_data(data, path):
    data.to_csv(path, index=False)


if __name__ == "__main__":
    config = read_yaml("config.yml")
    project_path = config["project_path"]
    n_regions = config['preprocessing']['amount_of_regions']
    n_std = config['preprocessing']['amount_of_std']

    df = read_data(os.path.join(project_path, *["data", "raw", "all_v2.csv"]))
    df = prep_date(df)
    df = clean_data(df)
    df = drop_regions(df, n_regions)
    df = filter_by_region(df, n_std, "price")
    df = filter_by_region(df, n_std, "area")
    df = make_time_features(df)
    df = convert_object_type(df)


    train, test = split_data(df, config['preprocessing']['train_size'])
    # train = reduce_mem_usage(train)
    # test = reduce_mem_usage(test)

    save_data(train, os.path.join(project_path, *["data", "clean", f"train_{n_regions}_{n_std}.csv"]))
    save_data(test, os.path.join(project_path, *["data", "clean", f"test_{n_regions}_{n_std}.csv"]))
