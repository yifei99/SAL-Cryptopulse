from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler


class DatasetStandard(Dataset):
    def __init__(self, data_path, dev_stage, window_setting):
        super(DatasetStandard, self).__init__()

        self.data_path = data_path
        assert dev_stage in [
            "train",
            "dev",
            "test",
        ], "Development stage must be train/dev/test"
        self.dev_stage = dev_stage

        # forecasting setting: [ob_len, pred_len]
        self.ob_len, self.pred_len = window_setting
        self.target_col = "close"
        self.scaler = StandardScaler()
        self.data = self.__read_data__()

    def __read_data__(self):
        # read csv
        df_raw = pd.read_csv(self.data_path)
        df_raw.columns = df_raw.columns.str.lower()
        df_raw["date"] = pd.to_datetime(df_raw["date"])
        df_raw = df_raw.sort_values(by=["date"])  # sort by date
        # split dataset to training/dev/test
        split = split_data(df_raw, self.ob_len)
        # preprocess features data and move target_col to the last column
        df_data = df_raw[
            [col for col in df_raw.columns if col not in ["date", self.target_col]]
            + [self.target_col]
        ]
        self.scaler.fit(df_data.iloc[split["train"][0] : split["train"][1]].values)
        df_data = df_data.iloc[split[self.dev_stage][0] : split[self.dev_stage][1]]
        df_data = self.scaler.transform(df_data.values)
        return df_data

    def __getitem__(self, index):
        # retrive the observation sequences
        s_begin, s_end = index, index + self.ob_len
        seq_x_data = self.data[s_begin:s_end]
        # retrieve the ground truth for the forecasting horizon
        f_begin, f_end = s_end, s_end + self.pred_len
        seq_y_data = self.data[f_begin:f_end]
        return seq_x_data, seq_y_data

    def __len__(self):
        return len(self.data) - self.ob_len - self.pred_len


_crypto_top20 = [
    "BTC",
    "ETH",
    "USDT",
    "BNB",
    "SOL",
    "STETH",
    "USDC",
    "XRP",
    "DOGE",
    "TON11419",
    "ADA",
    "SHIB",
    "AVAX",
    "WSTETH",
    "WETH",
    "DOT",
    "LINK",
    "WBTC",
    "TRX",
    "WTRX",
]
data_registry = {
    "{}-USD".format(crypto): {
        "dataset": DatasetStandard,
        "path": Path("dataset/") / "{}.{}-USD.csv".format(rank, crypto),
    }
    for rank, crypto in enumerate(_crypto_top20, start=1)
}


def split_data(data, ob_len, ratio=[0.7, 0.2]):
    len_data = data.shape[0]
    n_train = int(len_data * ratio[0])
    n_test = int(len_data * ratio[1])
    n_val = len_data - n_train - n_test
    return {
        "train": (0, n_train),
        "dev": (n_train - ob_len, n_train + n_val),
        "test": (len_data - n_test - ob_len, len_data),
    }


def data_provider(args, dev_stage):
    torch_dataset = data_registry[args.data]["dataset"]
    data_path = data_registry[args.data]["path"]
    dataset = torch_dataset(
        data_path=data_path,
        dev_stage=dev_stage,
        window_setting=[args.ob_len, args.pred_len],
    )
    print(" " * 4, dev_stage, len(dataset))
    if dev_stage in ["dev", "test"]:
        data_loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, drop_last=False
        )
    elif dev_stage == "train":
        data_loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
        )
    else:
        raise ValueError("Unknown development stage: {}".format(dev_stage))
    return data_loader
