import time
from corpus import WERDetails
from utils import HParam
#partition = "test"
#number_of_phonemes = 40
preprocessing = True

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import os
from figure_generator_2 import phoneme_barplot_2, phoneme_barplot_3



preprocessing = False
separation = True
if separation:
    files = glob("experiments/voicefilter_experiment/*_ss_result.txt")
else:
    files = glob("experiments/voicefilter_experiment/*_se_result.txt")
config = HParam("../configs/eng_espnet.yaml")



if preprocessing:
    dfs = list()
    for file in files:
        wer_details = WERDetails(file, skip_calculation=False, config=config)
        phoneme, other = wer_details.all_pers()

        dfs.append(pd.DataFrame(data=[other[1:]], columns=phoneme[1:], index=[file]))


    result = pd.concat(dfs, axis=0, join="outer")

    if separation:
        result.to_csv("csvs/separation_results_with_clean.csv")
    else:
        result.to_csv("csvs/enhancement_results_with_clean.csv")
else:

    if separation:
        df = pd.read_csv("../csvs/separation_results_with_clean.csv")
    else:
        df = pd.read_csv("../csvs/enhancement_results_with_clean.csv")
    df.index = df.iloc[:,0]
    df = df.iloc[:,1:]

    order = ["experiments/voicefilter_experiment/mixed_ss_result.txt",
              "experiments/voicefilter_experiment/dvec_ss_result.txt",
              "experiments/voicefilter_experiment/xvec_linear_ss_result.txt",
              "experiments/voicefilter_experiment/xvec_lstm_ss_result.txt",
              "experiments/voicefilter_experiment/xvec_lda_ss_result.txt",
              "experiments/voicefilter_experiment/gt_ss_result.txt"
              ]

    if not separation:
        new_order = [item.replace("_ss_","_se_") for item in order]
        order = new_order

    df = df.loc[order]
    df = df.dropna(axis=1,how='any',inplace=False)
    plot_data = df.values.T
    experiments = ["mixed","d-vec"," linear","x-vector LSTM","x-vector LDA","clean","mixed - dvec"]
    experiments = ["unenhanced","d-vector","x-vector linear","x-vector LSTM","x-vector LDA","clean","mixed - d-vector"]
    #experiments = ["xvec linear", "dvec", "xvec lstm", "mixed", "LDA"]

    if separation:
        phoneme_barplot_2(list(df.columns), per_ndarray=plot_data, experiment_names=experiments, filename="voicefilter_separation_2", conf=config, enhancement=False)
    else:
        phoneme_barplot_2(list(df.columns), per_ndarray=plot_data, experiment_names=experiments, filename="voicefilter_enhancement_2", conf=config, enhancement=True)

    #plt.plot(plot_data[1:,:])
    #plt.xlabel(df.cols)
    #plt.legend(plot_data[0,:])
    #ax = plt.gca()
    #x = np.arange(len(df.columns) - 1)
    #ax.set_xticks(x)

    #ax.set_xticklabels(df.columns[1:], rotation=45, fontsize=8)

    plt.show()
