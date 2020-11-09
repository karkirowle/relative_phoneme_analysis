import pandas as pd
import os
import numpy as np
root = "/home/boomkin/Documents/oral_cancer_asr_journal_paper/Archive/"

male = ["004","005","006","007","013","022","028","026","030","033","034"]
female = ["001","003","010","021","023","024","018"]
# go through each architecture

architectures = os.listdir(root)

# 5 is number of partitions, hardcoded, not nice, but this is life
male_wer = np.zeros((4,5))
female_wer = np.zeros((4,5))

for i,architecture in enumerate(architectures):
    partitions = os.listdir(os.path.join(root,architecture))

    for j,partition in enumerate(partitions):
        test = [p for p in os.listdir(os.path.join(root,architecture,partition)) if "test" in p]

        file = os.path.join(root,architecture,partition,test[0],"wer_details","per_spk")

        df = pd.read_csv(file,delim_whitespace=True)

        # filter id

        # select male word * error / total word

        df_sys = df[df["id"] == "sys"]
        df_male = df_sys[df_sys["SPEAKER"].isin(male)]
        male_wer[i,j] = np.sum(df_male["#WORD"].values * df_male["Err"].values) / np.sum(df_male["#WORD"].values)
        # select female word * error / total_word
        df_female = df_sys[df_sys["SPEAKER"].isin(female)]
        female_wer[i,j] = np.sum(df_female["#WORD"].values * df_female["Err"].values) / np.sum(df_female["#WORD"].values)




print(*architectures, sep='\t')
#print(architectures)
print(*np.mean(male_wer,axis=1),sep='\t')
print(*np.std(male_wer,axis=1),sep='\t')
print(*np.mean(female_wer,axis=1),sep='\t')
print(*np.std(female_wer,axis=1),sep='\t')
#print(male_wer)
#print(female_wer)
