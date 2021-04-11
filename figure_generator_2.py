import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import os
from typing import List
from corpus import WERDetails
from scipy.stats import pearsonr

# Sets to a nicer font type
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

def moa_to_phonemes(moa: str, config) -> List[str]:
    """

    Turns a Manner of Articulation feature into the corresponding set of phonemes

    :param moa:
    :return:
    """
    assert moa in config.phoneme.moa
    converter_table = pd.read_csv(os.path.join(os.path.dirname(__file__),config.phoneme.conversion_mapping))
    converter_table = converter_table[converter_table[config.phoneme.phoneme_name].notna()]
    converter_table = converter_table.fillna(0)

    converter_table = converter_table.set_index(converter_table[config.phoneme.phoneme_name])
    moa_filter = converter_table.loc[:, moa]
    moas = moa_filter.index[moa_filter == 1].values
    return moas

def round_to_n(x, n):
    return round(x, -int(np.floor(np.sign(x) * np.log10(abs(x)))) + n)

def articulatory_barplot(poa_afer: list, label_list: list, reference_label: list, filename: str, phoneme_count: np.ndarray, per: list):
    """
    Generates articulatory barplot

    :param poa_afer:
    :param label_list:
    :param reference_label:
    :return:
    """

    format_dict = {
        "width_fig": 3.14,
        "height_fig": 3.14 * 0.8, # 0.62
        "dpi": 100,
        "width": 0.13, # the width of the bars
        "fontsize": 7,
        "capsize": 1.5
    }

    num_architectures = len(label_list)
    x = np.arange(len(reference_label))

    mean_experiments = list()
    std_experiments = list()

    for experiment in poa_afer:
        selected_afer = np.zeros((len(experiment),len(reference_label)))
        for i, labelafer in enumerate(experiment):
            label, afer = labelafer

            idx = [label.index(lab) for lab in reference_label]
            afer = np.array(afer)[idx]
            selected_afer[i,:] = afer

        mean_afer = np.mean(selected_afer,axis=0)
        std_afer = np.std(selected_afer,axis=0)

        mean_experiments.append(mean_afer)
        std_experiments.append(std_afer)

    #width_logic = np.linspace(start=-format_dict["width"]*2,stop=format_dict["width"],num=num_architectures)

    # Phoeme counts

    wer_details = WERDetails("../experiments/baseline/wer_details/per_utt", skip_calculation=True)
    phoneme_labels = per[0][0][0]

    if "moa" in filename:
        af_labels = np.array([wer_details.phoneme_to_moa(phoneme_label) for phoneme_label in phoneme_labels])
    if "poa" in filename:
        af_labels = np.array([wer_details.phoneme_to_poa(phoneme_label) for phoneme_label in phoneme_labels])

    phoneme_counts = np.mean(phoneme_count,axis=0)
    af_counts = [round_to_n(np.sum(phoneme_counts[np.where(af_labels == af)]),2) for af in reference_label]

    fig = plt.figure(num=None, figsize=(format_dict["width_fig"], format_dict["height_fig"]),
                    dpi=format_dict["dpi"], facecolor='w', edgecolor='k')

    markers = ["v","^","<",">","x"]
    for i in range(num_architectures):
        # legend_props = {"elinewidth": 0.5}
        # plt.bar(x + format_dict["width"]/2 + width_logic[i],
        #             mean_experiments[i],format_dict["width"],
        #             label=label_list[i],yerr=std_experiments[i],
        #             capsize=format_dict["capsize"], error_kw=legend_props)
        plt.plot(x, mean_experiments[i], markersize=4, marker=markers[i])
        plt.fill_between(x, mean_experiments[i] - std_experiments[i], mean_experiments[i] + std_experiments[i],
                         alpha=0.2)
        r, p = pearsonr(af_counts, mean_experiments[i])
        print(label_list[i], " correlation btw data amount and performance", r, "p value", p)


        ax = plt.gca()
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='dashed',which="major")

    ax.axhline(y=0, color="black",linewidth=0.8)
    ax.set_xticks(x)

    ax.tick_params(axis='both', which='major', labelsize=format_dict["fontsize"], pad=1)

    reference_label_to_plot = [ref + " " + str(int(af_counts[i])) for i,ref in enumerate(reference_label)]
    ax.set_xticklabels(reference_label_to_plot, rotation=45, fontsize=format_dict["fontsize"])

    plt.ylabel("AFER (%)",fontsize=format_dict["fontsize"])

    plt.legend(label_list, fontsize=6,
               loc='upper center', bbox_to_anchor=(0.5, +1.32),
               fancybox=True, shadow=True, ncol=(num_architectures//2))
    plt.xlim([0,len(mean_experiments)])
    plt.ylim([20,80])
    plt.tight_layout(pad=0)
    fig.tight_layout(pad=0)
    fig.set_size_inches(format_dict["width_fig"], format_dict["height_fig"])

    current_dir = os.path.dirname(__file__)
    plt.savefig(os.path.join(current_dir, "figures/" + filename + ".pdf"), bbox_inches='tight', pad_inches=0.005)
    #plt.show()


def phoneme_barplot(per_list: list, mean_phoneme_count: list, experiment_names: list, filename: str):
    """
    Produces the phoneme barplot for the paper

    :param per_list: list containing the values for the PER
    :param mean_phoneme_count: list containing ints
    :param experiment_names: contains name of the experiments
    :param filename: filename (str) to use for saving
    :return:
    """

    reference_label = list()
    cat_lengths = list()
    cats = ["Vowel", "Plosive", "Nasal", "Fricative", "Affricates", "Approximant"]
    num_experiments = len(experiment_names)

    mean_phoneme_count = np.mean(mean_phoneme_count, axis=0)

    phonemes_under_hundred = [per_list[0][0][0][i] for i in range(len(mean_phoneme_count)) if mean_phoneme_count[i] < 100]
    # First, we filter the labels, the pre-determined list contains phonemes which will never occur
    # Second, we filter phonemes that have less than 100 Ns, because those are too unreliable for analysis
    phonemes_that_will_never_occur = ["AX","AXR","EM","EN","EL"]
    for moa in cats:
        phonemes = list(moa_to_phonemes(moa))
        for phoemes_to_remove in phonemes_that_will_never_occur:
            if phoemes_to_remove in phonemes: phonemes.remove(phoemes_to_remove)
        for phoneme_under_hundred in phonemes_under_hundred:
            if phoneme_under_hundred in phonemes: phonemes.remove(phoneme_under_hundred)

        # If no phonemes are left, drop the category from visualisation
        if len(phonemes) == 0:
            cats.remove(moa)
        else:
            cat_lengths.append(len(phonemes))
        reference_label.extend(phonemes)

    mean_experiments = list()
    std_experiments = list()
    for i,experiment in enumerate(per_list):
        selected_per = np.zeros((len(experiment),len(reference_label)))
        for j, labelper in enumerate(experiment):
            label, per = labelper

            idx = [label.index(lab) for lab in reference_label]
            per = np.array(per)[idx]
            selected_per[j,:] = per

            # At first step of nested, select the correspondingt mean phoneme counts
            if (i == 0) & (j == 0):
                selected_mean_phoneme_count = mean_phoneme_count[idx]

        mean_per = np.mean(selected_per,axis=0)
        std_per = np.std(selected_per,axis=0)

        mean_experiments.append(mean_per)
        std_experiments.append(std_per)

    format_dict = {
        "width_fig": 6.6,
        "height_fig": 3.14 * 0.75,
        "dpi": 100,
        "width": 0.40, # the width of the bars
        "fontsize": 10,
        "legend_fontsize": 7,
        "capsize": 1.5
    }

    fig_fontsize = 6  # 7

    fig = plt.figure(num=None, figsize=(format_dict["width_fig"], format_dict["height_fig"]),
                     dpi=format_dict["dpi"], facecolor='w', edgecolor='k')
    x = np.arange(len(mean_experiments[0]))

    markers = ["v","^","<",">","x"]
    print("Phoneme")
    for i in range(num_experiments):
        plt.plot(x,mean_experiments[i],markersize=4,marker=markers[i])
        plt.fill_between(x,mean_experiments[i] - std_experiments[i], mean_experiments[i] + std_experiments[i],alpha=0.2)
        r, p = pearsonr(mean_experiments[i],selected_mean_phoneme_count)
        print(experiment_names[i], " correlation btw data amount and performance", r, "p-value", p)
        print(experiment_names[i])

        # Sort best 5 phonemes
        best_phoneme_idx = np.argsort(mean_experiments[i], axis=0)[:5]
        print(np.array(reference_label)[best_phoneme_idx])

    ax = plt.gca()
    ax.set_xticklabels(reference_label, rotation=45, fontsize=fig_fontsize)

    ax.yaxis.grid(color='gray', linestyle='dashed', which="major")
    ax.xaxis.grid(color='gray', linestyle='dashed', which="major")
    ax.set_xticks(x)

    ax.set_xticklabels(["/" + x.lower() + "/" for x in reference_label], rotation=45, fontsize=format_dict["fontsize"])
    ax.tick_params(axis='both', which='major', labelsize=format_dict["fontsize"])

    plt.ylim([20,85])
    plt.xlim([0,len(reference_label)-1])
    plt.ylabel("PER (%)", fontsize=format_dict["fontsize"])

    plt.legend(experiment_names, fontsize=format_dict["legend_fontsize"],
               loc='upper center', bbox_to_anchor=(0.5, +1.3),
               fancybox=True, shadow=True, ncol=num_experiments)

    # This part produces the annotations in the bottom
    props = dict(connectionstyle='angle, angleA=180, angleB=90, rad=0', # the angle parameter controls the bent lines
                 arrowstyle='-',
                 shrinkA=1, # this controls the bounding box extent around the text
                 shrinkB=1, # same as above
                 color="black",
                 lw=1)


    accumulator = 0
    # for i,cat_length in enumerate(cat_lengths):
    #     cat = cats[i]
    #     ax.annotate(cat,
    #                 xy=(accumulator, -0),
    #                 xytext=(accumulator + cat_length / 2 - 1, -5.5),
    #                 annotation_clip=False,
    #                 fontsize=8,
    #                 arrowprops=props
    #     )
    #    ax.annotate(cat,
    #                xy=(accumulator + cat_length - 1, -0),
    #                xytext=(accumulator + cat_length / 2 - 1, -5.5),
    #                annotation_clip=False,
    #                fontsize=8,
    #                arrowprops=props
    #    )
    #    accumulator += cat_length

    plt.tight_layout()
    fig.tight_layout()
    fig.set_size_inches(format_dict["width_fig"], format_dict["height_fig"])

    current_dir = os.path.dirname(__file__)
    plt.savefig(os.path.join(current_dir,"figures/" + filename + ".pdf"),bbox_inches='tight',pad_inches = 0.005)

def phoneme_barplot_2(phoneme_list: list, per_ndarray: np.ndarray, experiment_names: list, filename: str, conf, enhancement=True):
    """

    :param phoneme_list: corresponding phonemes
    :param per_list: a per ndarray containg the PER results for each experiment (Number of Phonemes x Experiments)
    :param experiment_names: contains name of the experiments
    :param filename: filename (str) to use for saving
    :param config: config yaml
    :parma enhancement: enhancement config
    :return:
    """

    sorted_phoneme_label = list()
    moa_cat_lengths = list()
    moas = conf.phoneme.moa
    number_of_phonemes, num_experiments = per_ndarray.shape
    # First, we have to rearrange the phonemes by manner of articulation
    for moa in moas:

        phonemes_for_moa = list(moa_to_phonemes(moa,config=conf))

        # Perhaps only a subset of MoAs will actually occur in the dataset, so we select those
        occuring_phonemes_for_moa = [phoneme for phoneme in phonemes_for_moa if phoneme in phoneme_list]

        # If no phonemes are left, drop the category from visualisation
        if len(occuring_phonemes_for_moa) == 0:
            moas.remove(moa)
        else:
            # This variable stores how many phonemes are there for each MoA
            moa_cat_lengths.append(len(occuring_phonemes_for_moa))
        sorted_phoneme_label.extend(occuring_phonemes_for_moa)

    # sorted_phoneme_labels now contains the true order
    # First we acquire the sort_idx from the original list
    sort_idx = [phoneme_list.index(phoneme) for phoneme in sorted_phoneme_label]
    # Then, we just it so the PERs are also ordered
    sorted_per_ndarray = np.array(per_ndarray)[sort_idx,:]

    format_dict = {
        "width_fig": 6.6,
        "height_fig": 3.14 * 0.60, # 0.75
        "dpi": 100,
        "width": 0.40, # the width of the bars
        "fontsize": 10,
        "legend_fontsize": 6,
        "capsize": 1.5
    }

    fig_fontsize = 6  # 7

    fig = plt.figure(num=None, figsize=(format_dict["width_fig"], format_dict["height_fig"]),
                     dpi=format_dict["dpi"], facecolor='w', edgecolor='k')

    x = np.arange(number_of_phonemes)

    markers = ["v","^","<",">","x","v"]
    styles = ["solid","solid","solid","solid","solid","dashed"]

    for i in range(num_experiments):
        plt.plot(x,sorted_per_ndarray[:,i],markersize=4,marker=markers[i], linestyle=styles[i])

    # mixed - dvec
    plt.plot(x,sorted_per_ndarray[:,0] - sorted_per_ndarray[:,1],markersize=4,marker=4)
    ax = plt.gca()
    ax.set_xticklabels(sorted_phoneme_label, rotation=45, fontsize=fig_fontsize)

    ax.yaxis.grid(color='gray', linestyle='dashed', which="major")
    ax.xaxis.grid(color='gray', linestyle='dashed', which="major")
    ax.set_xticks(x)

    if enhancement:
        ax.set_xticklabels(["/" + x.lower() + "/" for x in sorted_phoneme_label], rotation=45, fontsize=format_dict["fontsize"])
    else:
        #ax.axes.xaxis.set_ticklabels([])
        #ax.axes.get_xaxis().set_visible(False)
        #ax.set_xticklabels(["/" + x.lower() + "/" for x in sorted_phoneme_label], rotation=45, fontsize=format_dict["fontsize"])
        ax.set_xticklabels([" " *  (len(x) + 2) for x in sorted_phoneme_label], rotation=45, fontsize=format_dict["fontsize"])
    if not enhancement:
        ax.tick_params(axis='y', which='major', labelsize=format_dict["fontsize"]-3)
    else:
        ax.tick_params(axis='y', which='major', labelsize=format_dict["fontsize"])

    #if not enhancement:
    #    plt.ylim([0,150])

    plt.xlim([0,len(sorted_phoneme_label)-1])

    plt.ylabel("IPER (%)", fontsize=format_dict["fontsize"])

    if not enhancement:
        plt.legend(experiment_names, fontsize=format_dict["legend_fontsize"],
               loc='upper center', bbox_to_anchor=(0.5, +1.3),
               fancybox=True, shadow=True, ncol=num_experiments+1)

    # This part produces the annotations in the bottom
    props = dict(connectionstyle='angle, angleA=180, angleB=90, rad=0', # the angle parameter controls the bent lines
                 arrowstyle='-',
                 shrinkA=1, # this controls the bounding box extent around the text
                 shrinkB=1, # same as above
                 color="black",
                 lw=1)

    # This part is responsible for the nice explanatory bars
    accumulator = 0

    moas = ["Affr" if x == "Affricates" else x for x in moas]
    moas = ["Apr" if x == "Approximant" else x for x in moas]

    if enhancement:
        y_offset = -8 #enhancement
    else:
        y_offset = -40
    y_offset_offset = - 10

    if enhancement:
        for i,moa_length in enumerate(moa_cat_lengths):
            moa = moas[i]
            fontsizes = [8,8,6,8,6,6]
            ax.annotate(moa,
                        xy=(accumulator, y_offset),
                        xytext=(accumulator + moa_length / 2 - 1, y_offset + y_offset_offset),
                        annotation_clip=False,
                        fontsize=fontsizes[i],
                        arrowprops=props
            )
            ax.annotate(moa,
                        xy=(accumulator + moa_length - 1, y_offset),
                        xytext=(accumulator + moa_length / 2 - 1, y_offset + y_offset_offset),
                        annotation_clip=False,
                        fontsize=fontsizes[i],
                        arrowprops=props
            )
            accumulator += moa_length

    plt.tight_layout()
    fig.tight_layout()
    fig.set_size_inches(format_dict["width_fig"], format_dict["height_fig"])

    current_dir = os.path.dirname(__file__)
    #plt.show()
    plt.savefig(os.path.join(current_dir,"figures/" + filename + ".pdf"),bbox_inches='tight',pad_inches = 0.005)

def phoneme_barplot_3(phoneme_list: list, per_ndarray: np.ndarray, experiment_names: list, filename: str, conf):
    """

    :param phoneme_list: corresponding phonemes
    :param per_list: a per ndarray containg the PER results for each experiment (Number of Phonemes x Experiments)
    :param experiment_names: contains name of the experiments
    :param filename: filename (str) to use for saving
    :param config: config yaml
    :return:
    """

    sorted_phoneme_label = list()
    moa_cat_lengths = list()
    moas = conf.phoneme.moa
    number_of_phonemes, num_experiments = per_ndarray.shape
    # First, we have to rearrange the phonemes by manner of articulation
    for moa in moas:

        phonemes_for_moa = list(moa_to_phonemes(moa,config=conf))

        # Perhaps only a subset of MoAs will actually occur in the dataset, so we select those
        occuring_phonemes_for_moa = [phoneme for phoneme in phonemes_for_moa if phoneme in phoneme_list]

        # If no phonemes are left, drop the category from visualisation
        if len(occuring_phonemes_for_moa) == 0:
            moas.remove(moa)
        else:
            # This variable stores how many phonemes are there for each MoA
            moa_cat_lengths.append(len(occuring_phonemes_for_moa))
        sorted_phoneme_label.extend(occuring_phonemes_for_moa)

    # sorted_phoneme_labels now contains the true order
    # First we acquire the sort_idx from the original list
    sort_idx = [phoneme_list.index(phoneme) for phoneme in sorted_phoneme_label]
    # Then, we just it so the PERs are also ordered
    sorted_per_ndarray = np.array(per_ndarray)[sort_idx,:]

    format_dict = {
        "width_fig": 6.6,
        "height_fig": 3.14 * 0.75,
        "dpi": 100,
        "width": 0.40, # the width of the bars
        "fontsize": 10,
        "legend_fontsize": 7,
        "capsize": 1.5
    }

    fig_fontsize = 6  # 7

    fig = plt.figure(num=None, figsize=(format_dict["width_fig"], format_dict["height_fig"]),
                     dpi=format_dict["dpi"], facecolor='w', edgecolor='k')

    x = np.arange(number_of_phonemes)

    markers = ["v","^","<",">","x","v"]

    for i in range(num_experiments):
        plt.plot(x,sorted_per_ndarray[:,i],markersize=4,marker=markers[i])

    #plt.plot(x,sorted_per_ndarray[:,3] - np.mean(sorted_per_ndarray[:,:3]))

    ax = plt.gca()
    ax.set_xticklabels(sorted_phoneme_label, rotation=45, fontsize=fig_fontsize)

    ax.yaxis.grid(color='gray', linestyle='dashed', which="major")
    ax.xaxis.grid(color='gray', linestyle='dashed', which="major")
    ax.set_xticks(x)

    ax.set_xticklabels(["/" + x + "/" for x in sorted_phoneme_label], rotation=45, fontsize=format_dict["fontsize"])
    ax.tick_params(axis='both', which='major', labelsize=format_dict["fontsize"])

    #plt.ylim([0,65]) 0,65 for elderly
    #plt.ylim([0,15])
    plt.xlim([0,len(sorted_phoneme_label)-1])
    plt.ylabel("PER (%)", fontsize=format_dict["fontsize"])

    plt.legend(experiment_names, fontsize=format_dict["legend_fontsize"],
               loc='upper center', bbox_to_anchor=(0.5, +1.3),
               fancybox=True, shadow=True, ncol=num_experiments+1)

    # This part produces the annotations in the bottom
    props = dict(connectionstyle='angle, angleA=180, angleB=90, rad=0', # the angle parameter controls the bent lines
                 arrowstyle='-',
                 shrinkA=1, # this controls the bounding box extent around the text
                 shrinkB=1, # same as above
                 color="black",
                 lw=1)

    # This part is responsible for the nice explanatory bars
    accumulator = 0

    moas = ["Affr" if x == "Affricates" else x for x in moas]
    moas = ["Apr" if x == "Approximant" else x for x in moas]

    #y_offset = -8 #enhancement
    y_offset = -35
    y_offset_offset = - 5
    for i,moa_length in enumerate(moa_cat_lengths):
        moa = moas[i]
        fontsizes = [8,8,6,8,6,6]
        ax.annotate(moa,
                    xy=(accumulator, y_offset),
                    xytext=(accumulator + moa_length / 2 - 1, y_offset + y_offset_offset),
                    annotation_clip=False,
                    fontsize=fontsizes[i],
                    arrowprops=props
        )
        ax.annotate(moa,
                    xy=(accumulator + moa_length - 1, y_offset),
                    xytext=(accumulator + moa_length / 2 - 1, y_offset + y_offset_offset),
                    annotation_clip=False,
                    fontsize=fontsizes[i],
                    arrowprops=props
        )
        accumulator += moa_length

    #plt.tight_layout()
    #fig.tight_layout()
    fig.set_size_inches(format_dict["width_fig"], format_dict["height_fig"])

    current_dir = os.path.dirname(__file__)
    #plt.show()
    plt.savefig(os.path.join(current_dir,"figures/" + filename + ".pdf"),bbox_inches='tight',pad_inches = 0.005)

def phoneme_barplot_normalised(phoneme_list: list, per_ndarray: np.ndarray, experiment_names: list, filename: str, conf,
                               select_idx: int, mean_idx: list):
    """

    :param phoneme_list: corresponding phonemes
    :param per_list: a per ndarray containg the PER results for each experiment (Number of Phonemes x Experiments)
    :param experiment_names: contains name of the experiments
    :param filename: filename (str) to use for saving
    :param config: config yaml
    :return:
    """

    sorted_phoneme_label = list()
    moa_cat_lengths = list()
    moas = conf.phoneme.moa
    number_of_phonemes, num_experiments = per_ndarray.shape
    # First, we have to rearrange the phonemes by manner of articulation
    for moa in moas:

        phonemes_for_moa = list(moa_to_phonemes(moa,config=conf))

        # Perhaps only a subset of MoAs will actually occur in the dataset, so we select those
        occuring_phonemes_for_moa = [phoneme for phoneme in phonemes_for_moa if phoneme in phoneme_list]

        # If no phonemes are left, drop the category from visualisation
        if len(occuring_phonemes_for_moa) == 0:
            moas.remove(moa)
        else:
            # This variable stores how many phonemes are there for each MoA
            moa_cat_lengths.append(len(occuring_phonemes_for_moa))
        sorted_phoneme_label.extend(occuring_phonemes_for_moa)

    # sorted_phoneme_labels now contains the true order
    # First we acquire the sort_idx from the original list
    sort_idx = [phoneme_list.index(phoneme) for phoneme in sorted_phoneme_label]
    # Then, we just it so the PERs are also ordered
    sorted_per_ndarray = np.array(per_ndarray)[sort_idx,:]

    format_dict = {
        "width_fig": 6.6,
        "height_fig": 3.14 * 0.75,
        "dpi": 100,
        "width": 0.40, # the width of the bars
        "fontsize": 10,
        "legend_fontsize": 7,
        "capsize": 1.5
    }

    fig_fontsize = 6  # 7

    fig = plt.figure(num=None, figsize=(format_dict["width_fig"], format_dict["height_fig"]),
                     dpi=format_dict["dpi"], facecolor='w', edgecolor='k')

    x = np.arange(number_of_phonemes)

    markers = ["v","^","<",">","x"]


    plt.plot(x,sorted_per_ndarray[:,select_idx] - np.mean(sorted_per_ndarray[:,mean_idx]),markersize=4,marker=markers[0])

    #plt.plot(x,sorted_per_ndarray[:,3] - np.mean(sorted_per_ndarray[:,:3]))

    ax = plt.gca()
    ax.set_xticklabels(sorted_phoneme_label, rotation=45, fontsize=fig_fontsize)

    ax.yaxis.grid(color='gray', linestyle='dashed', which="major")
    ax.xaxis.grid(color='gray', linestyle='dashed', which="major")
    ax.set_xticks(x)

    ax.set_xticklabels(["/" + x + "/" for x in sorted_phoneme_label], rotation=45, fontsize=format_dict["fontsize"])
    ax.tick_params(axis='both', which='major', labelsize=format_dict["fontsize"])

    #plt.ylim([0,65]) 0,65 for elderly
    #plt.ylim([0,15])
    plt.xlim([0,len(sorted_phoneme_label)-1])
    plt.ylabel("PER (%)", fontsize=format_dict["fontsize"])

    plt.legend(experiment_names, fontsize=format_dict["legend_fontsize"],
               loc='upper center', bbox_to_anchor=(0.5, +1.3),
               fancybox=True, shadow=True, ncol=num_experiments+1)

    # This part produces the annotations in the bottom
    props = dict(connectionstyle='angle, angleA=180, angleB=90, rad=0', # the angle parameter controls the bent lines
                 arrowstyle='-',
                 shrinkA=1, # this controls the bounding box extent around the text
                 shrinkB=1, # same as above
                 color="black",
                 lw=1)

    # This part is responsible for the nice explanatory bars
    accumulator = 0

    moas = ["Affr" if x == "Affricates" else x for x in moas]
    moas = ["Apr" if x == "Approximant" else x for x in moas]

    #y_offset = -8 #enhancement
    y_offset = -35
    y_offset_offset = - 5
    for i,moa_length in enumerate(moa_cat_lengths):
        moa = moas[i]
        fontsizes = [8,8,6,8,6,6]
        ax.annotate(moa,
                    xy=(accumulator, y_offset),
                    xytext=(accumulator + moa_length / 2 - 1, y_offset + y_offset_offset),
                    annotation_clip=False,
                    fontsize=fontsizes[i],
                    arrowprops=props
        )
        ax.annotate(moa,
                    xy=(accumulator + moa_length - 1, y_offset),
                    xytext=(accumulator + moa_length / 2 - 1, y_offset + y_offset_offset),
                    annotation_clip=False,
                    fontsize=fontsizes[i],
                    arrowprops=props
        )
        accumulator += moa_length

    plt.tight_layout()
    fig.tight_layout()
    fig.set_size_inches(format_dict["width_fig"], format_dict["height_fig"])

    current_dir = os.path.dirname(__file__)
    #plt.show()
    plt.savefig(os.path.join(current_dir,"figures/" + filename + ".pdf"),bbox_inches='tight',pad_inches = 0.005)



def visualise_confusion_matrices(confusion_matrices: list, experiment_name_list: list, filename: str) -> int:
    """
    Method which accepts a list of np.ndarrays containing the confusion matrices
    or a list of strings which contain the path to them

    Does not return anything, has the sideeffect of a visualisation
    :param confusion_matrices: list of np.ndarrays containing the confusion matrices
    :param experiment_name_list: a list of str containing the name of each experiment
    :param filename: prefix name to use when saving figure as pdf

    """

    def sum_confusion_matrices(confusion_matrices: list, reference_label: str) -> list:
        """
        Sums the confusion matrices according to the reference label
        :param confusion_matrices:
        :param reference_label:
        :return: a list containing the confusion matrix for each experiment
        """

        # This is to check that the labels are consistent when summing the columns and rows together
        consistency_label = confusion_matrices[0][0][1]

        cm_list = list()
        for i, experiment in enumerate(confusion_matrices):
            selected_cm = np.zeros((len(experiment), len(reference_label), len(reference_label)))
            for j, cmlabel in enumerate(experiment):
                cm, label = cmlabel
                assert label == consistency_label
                idx = [label.index(lab) for lab in reference_label]
                cm = np.array(cm)[idx, :]
                cm = cm[:, idx]
                selected_cm[j, :, :] = cm

            sum_cm = np.sum(selected_cm, axis=0)

            # Accuracy should show, how much gt was classified predicted class j so it should row wise
            # Axis=1 is row wise normalisation
            if i == 0:
                sum_cm = sum_cm / np.sum(sum_cm, axis=1) * 100
            else:
                sum_cm = sum_cm / np.sum(sum_cm, axis=1) * 100 - cm_list[0]

            cm_list.append(sum_cm)

        return cm_list

    # Visualisation parameters
    format_dict = {
        "width_fig": 6.6,
        "height_fig": 3.14 * 0.60,
        "dpi": 100,
        "width": 0.40, # the width of the bars
        "fontsize": 10,
        "legend_fontsize": 7,
        "capsize": 1.5
    }

    # First example will be reference label, insertion errors are removed from the visualisation
    # Pass by value is needed here because .remove() works in-place on the reference which causes problems with assert
    reference_label = confusion_matrices[0][0][1].copy()

    #if " " in reference_label:
    #    reference_label.remove(" ")



    # We are making the hard assumption that the labels will be ordered in the same way, this was checked previously
    # that this should be the case
    cm_list = sum_confusion_matrices(confusion_matrices,reference_label)

    fig = plt.figure(num=None, figsize=(format_dict["width_fig"], format_dict["height_fig"]),
                     dpi=format_dict["dpi"], facecolor='w', edgecolor='k')

    fig.patches.extend([plt.Rectangle((0.0, -0.03), 0.28, 0.9,
                                      fill=True, color='g', alpha=0.5, zorder=-1000,
                                      transform=fig.transFigure, figure=fig)])

    num_confusion_matrices = len(confusion_matrices)

    for i,confusion_matrix in enumerate(cm_list):
        plt.subplot(1,num_confusion_matrices,i+1)

        im = plt.imshow(confusion_matrix,cmap="PiYG")
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04) # this is a Stack Overflow copy paste for the sidebar
        cbar.ax.tick_params(labelsize=6)

        # The relative accuracy is scaled differently for the sake of argument
        if i == 0:
            plt.clim(-100, 100)
        else:
            plt.clim(-10,10)



        # This is needed so the numbers are right aligned
        for t in cbar.ax.get_yticklabels():
            t.set_horizontalalignment('right')
            # This (all-time function name competition winner) sets the padding from the colorbar
            t.set_x(4.5)

        if i == 0:
            ax = plt.gca()
            # We want to show all ticks...
            ax.set_xticks(np.arange(len(reference_label)))
            ax.set_yticks(np.arange(len(reference_label)))
            # ... and label them with the respective list entries

            x_labels = reference_label.copy()
            y_labels = reference_label.copy()
            x_labels[0] = "Deleted"
            y_labels[0] = "Inserted"

            ax.set_xticklabels(x_labels)
            ax.set_yticklabels(y_labels)

            # Rotate the tick labels and set their alignment.
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor", fontsize=6)
            plt.setp(ax.get_yticklabels(), fontsize=6)
        else:
            plt.tick_params(
                axis='both',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                left=False,
                right=False,
                labelleft=False,
                labelbottom=False)  # labels along the bottom edge are off

        # Setting an extra explanatory label for the percentages
        if i == 0:
            plt.xlabel("predicted")
            plt.ylabel("ground truth")
            ax = plt.gca()
            ax.patch.set_facecolor("#eafff5")
        if i == 4:
            plt.ylabel("% (relative) accuracy",fontsize=7,labelpad=27)
            ax = plt.gca()
            ax.yaxis.set_label_position("right")
        plt.title(experiment_name_list[i], fontsize=7)

    plt.subplots_adjust(wspace=0.40)
    fig.set_size_inches(format_dict["width_fig"], format_dict["height_fig"])

    # Draw a horizontal lines at those coordinates
    #if "moa" in filename:
        #line = plt.Line2D((.283, .283), (-0.04, 0.8), color="k", linewidth=1, linestyle="--")
        #plt.text(-34,11,"absolute",rotation="vertical",fontsize=7)
        #plt.text(-33,11,"relative",rotation="vertical",fontsize=7)
        #fig.add_artist(line)
    #if "poa" in filename:
        #line = plt.Line2D((.286, .286), (-0.04, 0.8), color="k", linewidth=1, linestyle="--")
        #plt.text(-48.2, 16, "absolute", rotation="vertical", fontsize=7)
        #plt.text(-46.6, 16, "relative", rotation="vertical", fontsize=7)
        #fig.add_artist(line)



    current_dir = os.path.dirname(__file__)
    #plt.show()

    fig = plt.gcf()
    ax = plt.gca()
    ax.set_facecolor("red")
    #plt.show()
    fig.savefig(os.path.join(current_dir,"figures/" + filename + ".pdf"),bbox_inches='tight',pad_inches = 0.005,
                facecolor=fig.get_facecolor(), transparent=True)

    return 1

