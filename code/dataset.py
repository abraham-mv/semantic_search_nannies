import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize

class NanniesDataframe:
    def __init__(self, 
                 input_path: str,
                 remove_duplicates: bool = True, 
                 tokenize_sentences: bool = True,
                 text_columns: list = ["short_blurb", "reasons", "about_me"], 
                 indicator_columns: list = ["name", "link", "date"]):
        self.path = input_path
        self.remove_duplicates = remove_duplicates
        self.tokenize_sentences = tokenize_sentences
        self.text_columns = text_columns
        self.indicator_columns = indicator_columns
        
        self.dataframe = pd.read_csv(input_path)[indicator_columns + text_columns]
        #super().__init__()
    
    def __len__(self) -> int:
        return len(self.dataframe)
    
    def tokenize(self: str):
        data_text = self.dataframe[self.text_columns].fillna(''). \
            apply(lambda x: x.str.replace('\r\n',"\n")). \
            apply(lambda x: x.str.replace(';',". "))

        sentences_full = []
        for _, row in data_text.iterrows(): 
            sentences = sent_tokenize(row.short_blurb)
            if len(row.reasons) > 0: sentences.extend(sent_tokenize(row.reasons))
            sentences.extend(sent_tokenize(row.about_me))
            sentences_full.append(sentences)
        return sentences_full
    
    
    def setup(self):
        if "date" in self.dataframe.columns:
            self.dataframe["date"] = pd.to_datetime(self.dataframe.date)
            self.dataframe = self.dataframe.sort_values("date", ascending=True)
        if self.remove_duplicates:
            self.dataframe = self.dataframe.drop_duplicates(subset=self.text_columns, keep='last')
        
        if self.tokenize_sentences: 
            sentences = self.tokenize()
            self.dataframe["sentences"] = [passages for passages in sentences]
            data_long = self.dataframe.drop(self.text_columns, axis=1). \
                        explode('sentences'). \
                        drop_duplicates(subset=["link", "sentences"], keep='last'). \
                        reset_index(drop=True).dropna()
            return data_long

if __name__ == "__main__":
    data = NanniesDataframe("data/input/nannies_profiles_canada.csv").setup()
    print(data)