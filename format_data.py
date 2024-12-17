import pandas as pd

reference_df = pd.read_csv('data-en-hi-de-fr 2.csv')

rows_to_add = []

labels = {'ham': 0, 'spam':1}

for idx, row in reference_df.iterrows():
    for condition in ['text', 'text_hi', 'text_de', 'text_fr']:
        rows_to_add.append({
            'textid': idx,
            'text': row[condition],
            'target': labels[row['labels']],
            'condition': condition
        })

df_final = pd.DataFrame(rows_to_add)

df_final.to_csv('data.tsv', sep='\t', index=False)

