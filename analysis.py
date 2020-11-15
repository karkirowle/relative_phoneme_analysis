from corpus import Corpus
from figure_generator import phoneme_barplot, articulatory_barplot
import os

import pandas as pd

# MODE 1: Add every test set example
# MODE 2: Add every test set speaker but ONCE, add first occurence
# MODE 3: Variant of mode 1, but calculate mean/std


def combine_df(df_list: list):
    """
    Merges the dataframes together from the different partitions and recalculates the differences and the relative
    error rates

    :param df_list: a list of dataframes from the different partitions
    :return:
    """
    df_sum = df_list[0][["phoneme","intentions","realisations"]].copy()
    df_sum.rename(columns={'realisations': 'realisations0', 'intentions': 'intentions0'},inplace=True)

    for i in range(1,len(df_list)):
        df_list[i].rename(columns={'realisations': 'realisations' + str(i), 'intentions': 'intentions' + str(i)},inplace=True)
        df_sum = df_sum.merge(df_list[i],how="outer",on="phoneme")

    for i in range(len(df_list)):

        df_sum["norm" + str(i)] = (df_sum["intentions" + str(i)] - df_sum["realisations" + str(i)]) / df_sum["intentions" + str(i)]

    df_sum["mean"] = df_sum[["intentions0","intentions1","intentions2","intentions3","intentions4"]].mean(axis=1,skipna=True)
    df_sum["norm_mean"] = df_sum[["norm0","norm1","norm2","norm3","norm4"]].mean(axis=1,skipna=True)
    df_sum["norm_std"] = df_sum[["norm0","norm1","norm2","norm3","norm4"]].std(axis=1,skipna=True)

    return df_sum

### Parameters that can be varied: used for consistency with previous experimental design

# no_stress: True -> ignores linguistic stress for the phoneme analysis. Articulatory analysis always ignores it.
no_stress = True
partition = "test"
single_partition = False
recordings_to_exclude = None
top_n_phoneme = 13
#recordings_to_exclude = ["30","33","34"]
#speakers_to_exclude = None

phoneme = []
moas = []
poas = []

for experiment in (os.listdir("experiments")):

    phoneme_dfs = []
    moa_dfs = []
    poa_dfs = []
    speaker_set = set()

    if single_partition:
        folds = [1]
    else:
        folds = range(1,6)

    for fold in folds:
        test_folder = [folder for folder in os.listdir(os.path.join("./experiments/",experiment,str(fold))) if partition in folder]
        wer_details = os.path.join("./experiments/",experiment,str(fold),test_folder[0],"wer_details","per_utt")

        corpus_1 = Corpus(wer_details)
        df, _ = corpus_1.get_dataframe(recordings_to_exclude=recordings_to_exclude, no_stress=no_stress)
        phoneme_dfs.append(df)
        poa, moa, _ = corpus_1.get_articulatory_dataframe(recordings_to_exclude=recordings_to_exclude)
        poa["phoneme"] = poa.index
        moa["phoneme"] = moa.index
        poa_dfs.append(poa)
        moa_dfs.append(moa)

    if single_partition:
        phoneme.append(phoneme_dfs[0])
        moas.append(moa_dfs[0])
        poas.append(poa_dfs[0])
    else:
        phoneme.append(combine_df(phoneme_dfs))
        moas.append(combine_df(moa_dfs))
        poas.append(combine_df(poa_dfs))

phoneme_barplot(phoneme,phoneme[4],
                ["Baseline","Baseline+OC","DNN AM retraining","FHVAE","fMLLR for AM retraining"],
                300,
                single_fold=single_partition,
                top_n_phonemes=top_n_phoneme)

articulatory_barplot(moas,["Baseline","Baseline+OC","DNN AM retraining","FHVAE","fMLLR"],"2010_11_10_moa", single_fold=single_partition)
articulatory_barplot(poas,["Baseline","Baseline+OC","DNN AM retraining","FHVAE","fMLLR"],"2010_11_10_poa", single_fold=single_partition)


