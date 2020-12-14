import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import itertools

# Sets to a nicer font type
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


def moa_to_phonemes(moa: str) -> list:
    assert moa in ["Vowel", "Plosive", "Nasal", "Fricative", "Affricates", "Approximant"]
    converter_table = pd.read_csv("PhoneSet_modified.csv")
    converter_table = converter_table[converter_table['ARPAbet'].notna()]
    converter_table = converter_table.fillna(0)

    # We consistently need to use uppercase throughout the code
    converter_table = converter_table.set_index(converter_table["ARPAbet"].str.upper())
    moa_filter = converter_table.loc[:, moa]
    moas = moa_filter.index[moa_filter == 1].values
    return moas

def articulatory_barplot(poa_afer: list, label_list: list, reference_label: list):

    format_dict = {
        "width_fig": 3.14,
        "height_fig": 3.14 * 0.8, # 0.62
        "dpi": 100,
        "width": 0.13, # the width of the bars
        "fontsize": 7
    }


    num_architectures = 5
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


    fig_fontsize = 6 # 7

    width_logic = np.linspace(start=-format_dict["width"]*2,stop=format_dict["width"],num=num_architectures)

    fig = plt.figure(num=None, figsize=(format_dict["width_fig"], format_dict["height_fig"]),
                    dpi=format_dict["dpi"], facecolor='w', edgecolor='k')

    for i in range(num_architectures):
        legend_props = {"elinewidth": 0.5}
        plt.bar(x + format_dict["width"]/2 + width_logic[i],
                    mean_experiments[i],format_dict["width"],
                    label=label_list[i],yerr=std_experiments[i],
                    capsize=1.0, error_kw=legend_props)

    ax = plt.gca()
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='dashed',which="major")

    ax.axhline(y=0, color="black",linewidth=0.8)
    ax.set_xticks(x)

    ax.tick_params(axis='both', which='major', labelsize=fig_fontsize, pad=1)
    ax.set_xticklabels(reference_label, rotation=45, fontsize=fig_fontsize)

    plt.ylabel("RAFER",fontsize=fig_fontsize)

    plt.legend(fontsize=6,
               loc='upper center', bbox_to_anchor=(0.5, +1.35),
               fancybox=True, shadow=True, ncol=(num_architectures//2))

    plt.tight_layout(pad=0)
    fig.tight_layout(pad=0)
    fig.set_size_inches(format_dict["width_fig"], format_dict["height_fig"])
    plt.show()

def phoneme_barplot(per_list: list, counts_holder: list, legend_val: list):

    reference_label = [moa_to_phonemes(moa) for moa in ["Vowel", "Plosive", "Nasal", "Fricative", "Affricates", "Approximant"]]
    reference_label = list(itertools.chain(*reference_label))
    reference_label.remove("AX")
    reference_label.remove("AXR")
    reference_label.remove("EM")
    reference_label.remove("EN")
    reference_label.remove("EL")

    counts_holder = np.mean(counts_holder,axis=0)
    #print(counts_holder)
    #print(len(counts_holder))
    #print(len(per_list[0][0][0]))
    under_hundred = [per_list[0][0][0][i] for i in range(len(counts_holder)) if counts_holder[i] < 100]

    for phoneme in under_hundred:
        if phoneme in reference_label:
            reference_label.remove(phoneme)
    #print(len(reference_label))
    #print(reference_label)
    num_experiments = 5
    mean_experiments = list()
    std_experiments = list()
    for experiment in per_list:
        selected_per = np.zeros((len(experiment),len(reference_label)))
        for i, labelper in enumerate(experiment):
            label, per = labelper
            #assert label == reference_label
            #print(label)
            idx = [label.index(lab) for lab in reference_label]
            #print(idx)
            per = np.array(per)[idx]
            #print(len(per))
            selected_per[i,:] = per

        mean_per = np.mean(selected_per,axis=0)
        std_per = np.std(selected_per,axis=0)

        mean_experiments.append(mean_per)
        std_experiments.append(std_per)

    format_dict = {
        "width_fig": 6.6,
        "height_fig": 3.14 * 0.60,
        "dpi": 100,
        "width": 0.40, # the width of the bars
        "fontsize": 10,
        "legend_fontsize": 7,
        "capsize": 1.5
    }

    fig_fontsize = 6  # 7
    width_logic = np.linspace(start=-format_dict["width"] * 2, stop=format_dict["width"], num=num_experiments)

    fig = plt.figure(num=None, figsize=(format_dict["width_fig"], format_dict["height_fig"]),
                     dpi=format_dict["dpi"], facecolor='w', edgecolor='k')
    x = np.arange(len(mean_experiments[0]))

    legend_props = {"elinewidth": 0.5}
    for i in range(num_experiments):
        plt.plot(x,mean_experiments[i],markersize=4,marker="x")
        plt.fill_between(x,mean_experiments[i] - std_experiments[i], mean_experiments[i] + std_experiments[i],alpha=0.2)
        #plt.scatter(x + 0.1 * i - 0.2,mean_experiments[i],s=6)
        #plt.bar(x + format_dict["width"] / 2 + width_logic[i],
                #mean_experiments[i], format_dict["width"],
                #label=reference_label, yerr=std_experiments[i],
                #capsize=1.0, error_kw=legend_props)
    ax = plt.gca()
    ax.set_xticklabels(reference_label, rotation=45, fontsize=fig_fontsize)

    ax.yaxis.grid(color='gray', linestyle='dashed', which="major")
    ax.xaxis.grid(color='gray', linestyle='dashed', which="major")
    #ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_xticks(x)

    ax.set_xticklabels(["/" + x.lower() + "/" for x in reference_label], rotation=45, fontsize=format_dict["fontsize"])
    ax.tick_params(axis='both', which='major', labelsize=format_dict["fontsize"])

    plt.ylim([0,85])
    plt.xlim([0,len(reference_label)-1])
    plt.ylabel("PER (%)", fontsize=format_dict["fontsize"])

    plt.legend(legend_val, fontsize=format_dict["legend_fontsize"],
               loc='upper center', bbox_to_anchor=(0.5, +1.3),
               fancybox=True, shadow=True, ncol=num_experiments)

    props = dict(connectionstyle='angle, angleA=180, angleB=90, rad=0',
                 arrowstyle='-',
                 shrinkA=1,
                 shrinkB=1,
                 color="black",
                 lw=1)

    cats = ["Vowel", "Plosive", "Nasal", "Fricative", "Approximant"]
    cat_lengths = [13,6,3,7,4]
    accumulator = 0
    for i,cat_length in enumerate(cat_lengths):
        cat = cats[i]
        ax.annotate(cat,
                    xy=(accumulator, -15),
                    xytext=(accumulator + cat_length / 2 - 1, -30.5),
                    annotation_clip=False,
                    fontsize=8,
                    arrowprops=props
        )
        ax.annotate(cat,
                    xy=(accumulator + cat_length - 1, -15),
                    xytext=(accumulator + cat_length / 2 - 1, -30.5),
                    annotation_clip=False,
                    fontsize=8,
                    arrowprops=props
        )
        accumulator += cat_length



    plt.tight_layout()
    fig.tight_layout()
    fig.set_size_inches(format_dict["width_fig"], format_dict["height_fig"])

    plt.show()

def visualise_confusion_matrices(confusion_matrices: list, legend_list: list) -> None:
    """

    Methods which accepts either a list of pd.Dataframes containing the confusion matrices
    or a list of strings which contain the path to them
    :param confusion_matrices:

    :return:
    """

    format_dict = {
        "width_fig": 6.6,
        "height_fig": 3.14 * 0.60,
        "dpi": 100,
        "width": 0.40, # the width of the bars
        "fontsize": 10,
        "legend_fontsize": 7,
        "capsize": 1.5
    }

    num_experiments = 5

    reference_label = confusion_matrices[0][0][1]
    reference_label.remove(" ")
    cm_list = list()
    for i, experiment in enumerate(confusion_matrices):
        selected_cm = np.zeros((len(experiment), len(reference_label), len(reference_label)))
        for j, cmlabel in enumerate(experiment):
            cm, label = cmlabel

            idx = [label.index(lab) for lab in reference_label]
            cm = np.array(cm)[idx,:]
            cm = cm[:,idx]
            selected_cm[j, :, :] = cm

        sum_cm = np.sum(selected_cm, axis=0)

        #assert np.sum(sum_cm,axis=0) == np.sum(sum_cm,axis=1)
        if i == 0:
            sum_cm = sum_cm / np.sum(sum_cm,axis=0) * 100
        else:
            sum_cm = sum_cm / np.sum(sum_cm, axis=0) * 100 - cm_list[0]

        cm_list.append(sum_cm)
    fig = plt.figure(num=None, figsize=(format_dict["width_fig"], format_dict["height_fig"]),
                     dpi=format_dict["dpi"], facecolor='w', edgecolor='k')


    num_confusion_matrices = len(confusion_matrices)

    for i,confusion_matrix in enumerate(cm_list):
        plt.subplot(1,num_confusion_matrices,i+1)

        im = plt.imshow(confusion_matrix,cmap="RdYlGn")
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=6)


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
            ax.set_xticklabels(reference_label)
            ax.set_yticklabels(reference_label)

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
        if i == 4:
            plt.ylabel("% (relative) accuracy",fontsize=7,labelpad=27)
            ax = plt.gca()
            ax.yaxis.set_label_position("right")
        plt.title(legend_list[i],fontsize=7)

    plt.subplots_adjust(wspace=0.40)
    fig.set_size_inches(format_dict["width_fig"], format_dict["height_fig"])
    #plt.savefig("figures/" + filename + ".pdf",bbox_inches='tight',pad_inches = 0.005)

    plt.show()