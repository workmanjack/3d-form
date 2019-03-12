import pandas as pd


class IndexedDataset(object):
    
    PCTILE_COL = None
    
    def __init__(self, df, index, pctile):
        self.index = index
        self.df = df
        self.pctile = pctile
        
    @classmethod
    def initFromIndex(cls, index, pctile=None):
        """
        Create a pandas df of the provided index file

        Args:
            index: str, path to thingi10k index file
            pctile: float, limit selection to at or below this pctile of num_vertices (0 to 1)

        Returns:
            pd.DataFrame
        """
        df = pd.read_csv(index)
        if pctile and cls.PCTILE_COL:
            df = dataframe_pctile_slice(df, cls.PCTILE_COL, pctile)
        return cls(df, index, pctile)

    def filter_to_just_one(self):
        self.df = self.df[:1]
        return
    
    def filter_by_id(self, filter_id):
        self.df = self.df[self.df[self.ID_COL].str.contains("{}".format(filter_id))]
        return

    def __getitem__(self, n):
        return self.df.loc[n]
    
    def __len__(self):
        return len(self.df)

    def __repr__(self):
        return '<IndexedDataset()>'
