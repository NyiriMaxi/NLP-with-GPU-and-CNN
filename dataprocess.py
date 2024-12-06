import pandas as pd
import numpy as np
import os as os

from tf_keras.callbacks import ModelCheckpoint,EarlyStopping

import matplotlib.pyplot as plt
import seaborn as sns




directory=os.getcwd()
#checkpoint
checkpoint_path="train_1/cp.ckpt"
checkpoint_dir=os.path.dirname(checkpoint_path)
es=EarlyStopping(monitor='val_accuracy',patience=3,restore_best_weights=True)


#checkpoint callback

checkpoint_cb=ModelCheckpoint(filepath=checkpoint_path, monitor='val_accuracy',save_best_only=True,verbose=1,mode='max')
callbackList=[checkpoint_cb,es]

#adatok betöltése


    

dataframe=pd.read_csv(directory+"/data/Language Detection.csv")
dataframe.sample(n=15,random_state=12345)


#A hindi szöveg hiánya miatt feldarabolom több részre

hindi_text=dataframe[dataframe['Language']=='Hindi'][['Text','Language']]

split_data=[]
split_len=20
for _, row in hindi_text.iterrows():
    text = row['Text']
    label = row['Language']
    if(isinstance(text,str) and len(text)>0):
        split_size = len(text) // split_len  # Az egyes részek hossza
        for i in range(split_len):
            start_index = i * split_size
            if i == split_len-1: 
                end_index = len(text)
            else:
                end_index = start_index + split_size
            split_data.append({'Text': text[start_index:end_index], 'Language': label})

split_df=pd.DataFrame(split_data)   

updated_dataframe=pd.concat([dataframe, split_df], ignore_index=True)


if(os.path.exists(directory+"/data/updated_data.csv")):
    print(f'A fájl már létezik: {directory+"/data/updated_data.csv"}')
else:
    updated_dataframe.to_csv('updated_data.csv', index=False)
    print(f'A fájl mentve lett: {directory+"/data/updated_data.csv"}')

languages=updated_dataframe["Language"].value_counts().reset_index()
languages.columns=["Nyelv","Előfordulás"]
print(languages)

#Visualizáció
plt.figure(figsize=(10, 6))
sns.barplot(x='Előfordulás', y='Nyelv', data=languages, palette='Paired')
plt.xlabel('Előfordulás')
plt.ylabel('Nyelv')
plt.title('A nyelvek szétoszlása')
plt.tight_layout()
plt.show()


