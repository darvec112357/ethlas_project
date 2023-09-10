import pandas as pd
import numpy as np

test_df = pd.read_csv('Phishing_Email.csv',encoding='latin-1')
test_df = test_df.drop(columns=['Unnamed: 0'])
test_df = test_df.drop_duplicates()
test_df = test_df.rename(columns = {'Email Text':'text','Email Type':'label'})
test_df.loc[test_df["label"] == "Phishing Email", "label"] = 0
test_df.loc[test_df["label"] == "Safe Email", "label"] = 1
test_df['text'].replace('', np.nan, inplace=True)
test_df.dropna(subset=['text'], inplace=True)
test_df.to_csv('train.csv',index=False)
