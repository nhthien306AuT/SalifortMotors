import pandas as pd

class data_cleaner:
    def __init__(self, loader):
        self.df = loader.df
        self.report = {} 

    def handle_nulls(self):
        before = len(self.df)
        self.df = self.df.dropna()
        dropped = before - len(self.df)
        self.report['dropped_nulls'] = dropped
        print(f"✅ Dropped {dropped} rows containing nulls")
        return self

    def remove_duplicates(self):
        before = len(self.df)
        self.df = self.df.drop_duplicates()
        removed = before - len(self.df)
        self.report['removed_duplicates'] = removed
        print(f"✅ Removed {removed} duplicate rows")
        return self
    
    def convert_datetime(self, columns, date_format=None):
        for col in columns:
            before_nulls = self.df[col].isna().sum()
            self.df[col] = pd.to_datetime(self.df[col], format=date_format, errors='coerce')
            after_nulls = self.df[col].isna().sum()
            self.report[f'datetime_{col}'] = after_nulls - before_nulls
            if after_nulls - before_nulls > 0:
                print(f"✅ Column '{col}': {after_nulls - before_nulls} values converted to NaT")
        return self
    
    def trim(self):
        for col in self.df.select_dtypes(include='object').columns:
            before_changes = (self.df[col] != self.df[col].str.strip()).sum()
            self.df[col] = self.df[col].str.strip()
            self.report[f'trim_{col}'] = before_changes
            print(f"✅ Column '{col}': trimmed {before_changes} values")
        return self

    def standardize(self):
        for col in self.df.select_dtypes(include='object').columns:
            before_changes = (self.df[col] != self.df[col].str.lower()).sum()
            self.df[col] = self.df[col].str.lower()
            if before_changes > 0:
                self.report[f'standardize_{col}'] = before_changes
                print(f"✅ Column '{col}': standardized {before_changes} values")
        
        old_cols = self.df.columns.tolist()
        self.df.columns = [c.lower() for c in self.df.columns]
        col_changes = sum(1 for o, n in zip(old_cols, self.df.columns) if o != n)
        print(f"✅ Standardized {col_changes} column names to lowercase")
        return self
