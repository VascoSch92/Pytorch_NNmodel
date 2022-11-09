# Packages and modules
import numpy as np
from torch.utils.data import Dataset, random_split


class TabularDataset(Dataset):

    def __init__(self, data, cat_cols=None, output_col=None):
        """
        Characterizes a Dataset for PyTorch

        Parameters
        ----------
        data: pandas dataframe
            It contains all the continuous, categorical and output columns to be used.
        cat_cols: list of strings
            Names of the categorical columns in the dataframe. These columns will be passed
            trought the embedding layers in the model. They must be labeled encoded beforehand.
        output_col: string
            The name of the output variable column in the data provided
        """
        
        # Number of samples
        self.n = data.shape[0]

        # Output part of the dataframe
        if output_col:
            self.y = data[output_col].astype(np.float32).values.reshape(-1, 1)
        else:
            self.y = np.zeros((self.n, 1))

        # Selecting the categorical and continuous columns
        self.cat_cols = cat_cols if cat_cols else []
        self.cont_cols = [col for col in data.columns if col not in self.cat_cols + [output_col]]

        # Continuous part of the dataframe
        if self.cont_cols:
            self.cont_X = data[self.cont_cols].astype(np.float32).values
        else:
            self.cont_X = np.zeros((self.n, 1))

        # Categorical part of the dataframe
        if self.cat_cols:
            self.cat_X = data[cat_cols].astype(np.int64).values
        else:
            self.cat_X = np.zeros((self.n, 1))

    def __len__(self):
        """
        Returns the total number of samples.
        """
        return self.n

    def __getitem__(self, idx):
        """
        Generates one sample of data.
        """
        return {"target": self.y[idx], "cont_data": self.cont_X[idx], "cat_data": self.cat_X[idx]}

    def get_splits(self, split=0.33):
        """
        Get indexes for train and validation rows
        """
        # Determines size
        test_size = round(split * self.n)
        train_size = self.n - test_size

        # calculate the split
        return random_split(self, [train_size, test_size])
