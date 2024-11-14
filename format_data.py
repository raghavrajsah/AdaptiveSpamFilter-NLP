import pandas as pd

column_names = ["labels", "text", "text_hi", "text_de", "text_fr"]
df = pd.read_csv('data-en-hi-de-fr 2.csv', names=column_names, skiprows=1)

label_mapping = {'ham': 0, 'spam': 1}

df_final = pd.DataFrame(columns=['textid', 'text', 'label'])

idx = 0
rows_to_add = []
for _, row in df.iterrows():
    rows_to_add.append({'textid': idx, 'text': row['text'], 'label': label_mapping[row['labels']]})
    idx += 1
    rows_to_add.append({'textid': idx, 'text': row['text_hi'], 'label': label_mapping[row['labels']]})
    idx += 1
    rows_to_add.append({'textid': idx, 'text': row['text_de'], 'label': label_mapping[row['labels']]})
    idx += 1
    rows_to_add.append({'textid': idx, 'text': row['text_fr'], 'label': label_mapping[row['labels']]})
    idx += 1

df_final = pd.concat([df_final, pd.DataFrame(rows_to_add)], ignore_index=True)

df_final.to_csv('multi_lingual.tsv', sep='\t', index=False)
