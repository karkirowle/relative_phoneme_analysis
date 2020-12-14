from corpus import Corpus
from figure_generator import phoneme_barplot, articulatory_barplot, visualise_confusion_matrices
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


def combine_confusion_matrix(confusion_matrix_list: list) -> pd.DataFrame:

    reference = confusion_matrix_list[0]

    for confusion_matrix in range(1,len(confusion_matrix_list)):
        reference = reference.add(confusion_matrix)

    return reference


### Parameters that can be varied: used for consistency with previous experimental design

# no_stress: True -> ignores linguistic stress for the phoneme analysis. Articulatory analysis always ignores it.
no_stress = True
partition = "test"
single_partition = False
recordings_to_exclude = None
top_n_phoneme = 17
number_of_occurence = 100

# This takes long so it is optional, otherwise loads csv
calculate_confusion_matrix = False
calculate_af_confusion_matrix = False
#recordings_to_exclude = ["30","33","34"]
#speakers_to_exclude = None

phoneme = []
moas = []
poas = []
confusion_matrices = []
moa_confusion_matrices = []
poa_confusion_matrices = []
for experiment in (os.listdir("experiments")):

    phoneme_dfs = []
    moa_dfs = []
    poa_dfs = []
    confusion_dfs = []
    moa_confusion_dfs = []
    poa_confusion_dfs = []
    speaker_set = set()

    if single_partition:
        folds = [1]
    else:
        folds = range(1,6)

    for fold in folds:
        test_folder = [folder for folder in os.listdir(os.path.join("./experiments/",experiment,str(fold))) if partition in folder]
        wer_details = os.path.join("./experiments/",experiment,str(fold),test_folder[0],"wer_details","per_utt")
        ops = os.path.join("./experiments/",experiment,str(fold),test_folder[0],"wer_details","ops")

        corpus_1 = Corpus(wer_details,ops)
        df, _ = corpus_1.get_dataframe(recordings_to_exclude=recordings_to_exclude, no_stress=no_stress)
        phoneme_dfs.append(df)
        poa, moa, _ = corpus_1.get_articulatory_dataframe(recordings_to_exclude=recordings_to_exclude)
        poa["phoneme"] = poa.index
        moa["phoneme"] = moa.index
        poa_dfs.append(poa)
        moa_dfs.append(moa)
        if calculate_confusion_matrix:
            confusion_matrix = corpus_1.get_confusion_matrix()
            confusion_dfs.append(confusion_matrix)
        if calculate_af_confusion_matrix:
            poa_cm, moa_cm = corpus_1.get_articulatory_confusion()
            poa_confusion_dfs.append(poa_cm)
            moa_confusion_dfs.append(moa_cm)

    if single_partition:
        phoneme.append(phoneme_dfs[0])
        poas.append(poa_dfs[0])
        moas.append(moa_dfs[0])

        if calculate_confusion_matrix:
            confusion_matrices.append(confusion_matrix)
        if calculate_af_confusion_matrix:
            poa_confusion_matrices.append(poa_confusion_dfs)
            moa_confusion_matrices.append(moa_confusion_dfs)
    else:
        phoneme.append(combine_df(phoneme_dfs))
        poas.append(combine_df(poa_dfs))
        moas.append(combine_df(moa_dfs))

        if calculate_confusion_matrix:
            confusion_matrices.append(combine_confusion_matrix(confusion_dfs))
        if calculate_af_confusion_matrix:
            poa_confusion_matrices.append((combine_confusion_matrix(poa_confusion_dfs)))
            moa_confusion_matrices.append((combine_confusion_matrix(moa_confusion_dfs)))


if calculate_confusion_matrix:
    confusion_matrices[0].to_csv("baseline_conf.csv")
    confusion_matrices[1].to_csv("baseline_plus_oc_conf.csv")
    confusion_matrices[2].to_csv("dnn_am_conf.csv")
    confusion_matrices[3].to_csv("fhvae_conf.csv")
    confusion_matrices[4].to_csv("fmllr_conf.csv")
    visualise_confusion_matrices(confusion_matrices)
else:
    print("sajt")
    #visualise_confusion_matrices(["conf_baseline.csv","baseline_plus_oc_conf.csv","dnn_am_conf.csv","fhvae_conf.csv", "fmllr_conf.csv"])

if calculate_af_confusion_matrix:
    poa_confusion_matrices[0].to_csv("csvs/poa_baseline_conf.csv")
    poa_confusion_matrices[1].to_csv("csvs/poa_baseline_plus_oc_conf.csv")
    poa_confusion_matrices[2].to_csv("csvs/poa_dnn_am_conf.csv")
    poa_confusion_matrices[3].to_csv("csvs/poa_fhvae_conf.csv")
    poa_confusion_matrices[4].to_csv("csvs/poa_fmllr_conf.csv")
    moa_confusion_matrices[0].to_csv("csvs/moa_baseline_conf.csv")
    moa_confusion_matrices[1].to_csv("csvs/moa_baseline_plus_oc_conf.csv")
    moa_confusion_matrices[2].to_csv("csvs/moa_dnn_am_conf.csv")
    moa_confusion_matrices[3].to_csv("csvs/moa_fhvae_conf.csv")
    moa_confusion_matrices[4].to_csv("csvs/moa_fmllr_conf.csv")
    visualise_confusion_matrices(poa_confusion_matrices)
    visualise_confusion_matrices(moa_confusion_matrices)
else:
    visualise_confusion_matrices(["csvs/poa_baseline_conf.csv",
                                  "csvs/poa_baseline_plus_oc_conf.csv",
                                  "csvs/poa_dnn_am_conf.csv",
                                  "csvs/poa_fhvae_conf.csv",
                                  "csvs/poa_fmllr_conf.csv"],
                                 False,
                                 ["Baseline","Baseline+OC","DNN for AM retraining", "FHVAE", "fMLLR for AM retraining"],
                                 "poa_confusion")

    visualise_confusion_matrices(["csvs/moa_baseline_conf.csv",
                                  "csvs/moa_baseline_plus_oc_conf.csv",
                                  "csvs/moa_dnn_am_conf.csv",
                                  "csvs/moa_fhvae_conf.csv",
                                  "csvs/moa_fmllr_conf.csv"],
                                 False,
                                 ["Baseline", "Baseline+OC", "DNN for AM retraining", "FHVAE",
                                  "fMLLR for AM retraining"],
                                 "moa_confusion")

phoneme[4].to_csv("fmlrr_result.csv")

phoneme_barplot(phoneme,phoneme[4],
                ["Baseline","Baseline+OC","DNN AM retraining","FHVAE","fMLLR for AM retraining"],
                number_of_occurence=number_of_occurence,
                single_fold=single_partition,
                top_n_phonemes=top_n_phoneme)

articulatory_barplot(moas,["Baseline","Baseline+OC","DNN AM retraining","FHVAE","fMLLR"],
                     "2020_11_10_moa", single_fold=single_partition, poa=False)

articulatory_barplot(poas,["Baseline","Baseline+OC","DNN AM retraining","FHVAE","fMLLR"],
                     "2020_11_10_poa", single_fold=single_partition, poa=True)


