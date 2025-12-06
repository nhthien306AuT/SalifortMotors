import pandas as pd

class data_loader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df= None

    def load_csv(self):
        self.df = pd.read_csv(self.file_path)
        print("Data loaded successfully.")
        print(self.df.info())
        return self