
import time
from corpus import WERDetails
from utils import HParam, CharConverter
#partition = "test"
#number_of_phonemes = 40
preprocessing = True

config = HParam("../configs/dutch.yaml")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import os
from figure_generator_2 import phoneme_barplot_2, phoneme_barplot_3, phoneme_barplot_normalised
import numpy as np

def top_3_for_each():
    df = pd.read_csv("../csvs/jasmin_analysis_results.csv")
    df.index = df.iloc[:, 0]
    df = df.iloc[:, 1:]
    del df["[SPN]"]

    df_count = pd.read_csv("../csvs/count_jasmin_analysis_results.csv")
    df_count.index = df_count.iloc[:, 0]
    df_count = df_count.iloc[:, 1:]
    del df_count["[SPN]"]
    #df_count = np.nan_to_num(df_count)
    df_count_np = df_count.to_numpy().astype(np.float32)
    #astype(np.float32)
    #df_count["sum"] = np.nan_to_num(df_count_np).sum(axis=1)
    #print("")
    #df_count_ratio = df_count.div(df_count["sum"], axis=0)
    num = 5

    converter = CharConverter("uni", "tipa")

    for i in range(len(df.index)):
        #print(df[])
        #print(df.index[i])
        #df.index[i] = df.index[i].replace("group1","DC") # dutch child
        #df.index[i] = df.index[i].replace("group2","DT") # dutch teen
        #df.index[i] = df.index[i].replace("group3","NNT") # non native teen
        #df.index[i] = df.index[i].replace("group4","NNA") # non native adult
        #df.index[i] = df.index[i].replace("group5","DOA") # dutch old

        if "/vl/read/" in df.index[i]:
            continue
        #print(df.index[i])
        name = df.index[i].split("/")[-2]
        name = name.replace("group1","DC")
        name = name.replace("group2","DT")
        name = name.replace("group3","NNT")
        name = name.replace("group4","NNA")
        name = name.replace("group5","DOA")
        print(df.index[i].split("/")[-4] + "/" + name,end="\t")
        # For the sake of sorting, we don't care about nans, so nans are zeroed
        row_preprocessed = np.nan_to_num(df.iloc[i,:].values)
        row_preprocessed[df_count.iloc[i,:].values < 50] = 0
        # Descending order: do it from backwards
        idx = np.argsort(row_preprocessed)[::-1][:num]


        #top_phonemes= list(df.columns[idx])
        print([converter.convert(str(df.columns[idx[j]])) + " PER:" + str(np.round(df.iloc[i,idx[j]],2)) + " count:" + str(df_count.iloc[i,idx[j]]) for j in range(num)])

    print("")

def elderly():
    df = pd.read_csv("../csvs/jasmin_analysis_results.csv")

    df.index = df.iloc[:, 0]
    df = df.iloc[:, 1:]

    root = "/home/boomkin/jasmin_analysis_phoneme/nl/read/"
    jasmin_order = ["group5_N1",
                    "group5_N2",
                    "group5_N3",
                    "group5_N4"]
    jasmin_order = [os.path.join(root, file, "per_utt") for file in jasmin_order]
    jasmin_order.append("/home/boomkin/jasmin_analysis_phoneme/control_bn/per_utt")

    df = df.loc[jasmin_order]
    del df["[SPN]"]
    df = df.dropna(axis=1, how='any', inplace=False)
    plot_data = df.values.T

    experiments = ["N1", "N2", "N3", "N4","control"]
    phoneme_barplot_3(list(df.columns), per_ndarray=plot_data, experiment_names=experiments,
                      filename="elderly_by_region_jasmin", conf=config)
    experiments = ["N4 - mean(N1,N2,N3,control)"]

    phoneme_barplot_normalised(list(df.columns), per_ndarray=plot_data, experiment_names=experiments,
                      filename="elderly_by_region_jasmin_normalised", conf=config, select_idx=3, mean_idx=[0,1,2,4])
    plt.show()



def language():
    df = pd.read_csv("../csvs/jasmin_analysis_results.csv")

    df.index = df.iloc[:, 0]
    df = df.iloc[:, 1:]

    root = "/home/boomkin/jasmin_analysis_phoneme/nl/read/"
    jasmin_order = ["group4_A1",
                    "group4_A2",
                    "group4_B1"]
    jasmin_order = [os.path.join(root, file, "per_utt") for file in jasmin_order]
    jasmin_order.append("/home/boomkin/jasmin_analysis_phoneme/control_bn/per_utt")

    df = df.loc[jasmin_order]
    del df["[SPN]"]
    df = df.dropna(axis=1, how='any', inplace=False)
    plot_data = df.values.T

    experiments = ["CEF A1", "CEF A2", "CEF B1", "native control"]
    phoneme_barplot_3(list(df.columns), per_ndarray=plot_data, experiment_names=experiments,
                      filename="language_acquisition", conf=config)
    experiments = ["CEF A1 - control"]
    phoneme_barplot_normalised(list(df.columns), per_ndarray=plot_data, experiment_names=experiments,
                      filename="language_aq_normalised", conf=config, select_idx=0, mean_idx=[3])
    #plt.show()

def age():
    df = pd.read_csv("../csvs/jasmin_analysis_results.csv")

    df.index = df.iloc[:, 0]
    df = df.iloc[:, 1:]

    root = "/home/boomkin/jasmin_analysis_phoneme/nl/read/"
    jasmin_order = ["group1",
                    "group2",
                    "group5"]
    jasmin_order = [os.path.join(root, file, "per_utt") for file in jasmin_order]
    jasmin_order.append("/home/boomkin/jasmin_analysis_phoneme/control_bn/per_utt")
    #jasmin_order.append("/home/boomkin/jasmin_analysis_phoneme/control_cts/per_utt")

    print(jasmin_order)
    df = df.loc[jasmin_order]
    del df["[SPN]"]
    df = df.dropna(axis=1, how='any', inplace=False)
    plot_data = df.values.T

    experiments = ["native 7-11 years", "native 12-16", "native 65+","BN control"]
    phoneme_barplot_3(list(df.columns), per_ndarray=plot_data, experiment_names=experiments,
                      filename="age_differences", conf=config)

    plt.show()

def merge_region_transcripts():

    regions = ["N1","N2","N3","N4"]
    for region in regions:
        files = glob("/home/boomkin/jasmin_analysis_phoneme/nl/read/group*_" + region + "/per_utt")

        # Merges all TXT
        os.mkdir("/home/boomkin/jasmin_analysis_phoneme/nl/read/" + region)
        with open("/home/boomkin/jasmin_analysis_phoneme/nl/read/" + region + "/per_utt", "w") as outfile:

            for filename in files:
                with open(filename) as infile:
                    contents = infile.read()

                    outfile.write(contents)

def merge_native_transcripts():
    files = ["/home/boomkin/jasmin_analysis_phoneme/nl/read/group1/per_utt",
             "/home/boomkin/jasmin_analysis_phoneme/nl/read/group2/per_utt",
             "/home/boomkin/jasmin_analysis_phoneme/nl/read/group5/per_utt"]

    # Merges all TXT
    os.makedirs("/home/boomkin/jasmin_analysis_phoneme/nl/read/native/", exist_ok=True)
    with open("/home/boomkin/jasmin_analysis_phoneme/nl/read/native/per_utt", "w") as outfile:
        for filename in files:
            with open(filename) as infile:
                contents = infile.read()

                outfile.write(contents)

def merge_vl_transcripts():
    files = ["/home/boomkin/jasmin_analysis_phoneme/vl/read/group1/per_utt",
             "/home/boomkin/jasmin_analysis_phoneme/vl/read/group2/per_utt",
             "/home/boomkin/jasmin_analysis_phoneme/vl/read/group5/per_utt"]

    # Merges all TXT
    os.mkdir("/home/boomkin/jasmin_analysis_phoneme/vl/read/vl_all/")
    with open("/home/boomkin/jasmin_analysis_phoneme/vl/read/vl_all/per_utt", "w") as outfile:
        for filename in files:
            with open(filename) as infile:
                contents = infile.read()

                outfile.write(contents)

if __name__ == '__main__':
    #merge_native_transcripts()
    #merge_vl_transcripts()
    #merge_region_transcripts()
    top_3_for_each()
    #language()
    #age()
    #elderly()