from pathlib import Path
import pandas as pd
import numpy as np
root=Path(__file__).resolve().parent
df=pd.read_csv(root/'data/raw/bank-additional-full.csv',sep=';')
split=int(len(df)*0.60)
(root/'data/base').mkdir(parents=True,exist_ok=True)
(root/'data/incoming').mkdir(parents=True,exist_ok=True)
df.iloc[:split].to_csv(root/'data/base/base.csv',sep=';',index=False)
for i,batch in enumerate(np.array_split(df.iloc[split:],4),1):
    batch.to_csv(root/'data/incoming'/f'batch_{i:02}.csv',sep=';',index=False)
print('Исходная часть:',split,'порции:',[len(b) for b in np.array_split(df.iloc[split:],4)])
