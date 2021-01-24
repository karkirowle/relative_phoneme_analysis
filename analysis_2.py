import os
from corpus_2 import WERDetails
import pickle
import time
import numpy as np
from figure_generator_2 import phoneme_barplot, articulatory_barplot, visualise_confusion_matrices
partition = "test"
number_of_phonemes = 40
preprocessing = False

if preprocessing:
    per, poa_afer, moa_afer, poa_cm, moa_cm = [[],[],[],[],[]]
    experiment_folders = sorted(os.listdir("experiments"))
    print(experiment_folders)
    # Purpose of this is to just store pure phoneme count
    phoneme_count_per_fold = np.zeros((len(experiment_folders), number_of_phonemes))
    for i, experiment in enumerate(experiment_folders):

        per_per_experiment, poa_afer_per_experiment, moa_afer_per_experiment,\
            poa_cm_per_experiment, moa_cm_per_experiment = [[],[],[],[],[]]

        for fold in range(1,6):
            t = time.time()
            test_folder = [folder for folder in os.listdir(os.path.join("./experiments/",experiment,str(fold)))
                           if partition in folder]

            wer_details = os.path.join("./experiments/",experiment,str(fold),test_folder[0],"wer_details","per_utt")

            corpus = WERDetails(wer_details)

            per_per_experiment.append(corpus.all_pers())
            poa_afer_per_experiment.append(corpus.all_poa_afers())
            moa_afer_per_experiment.append(corpus.all_moa_afers())

            poa_cm_per_experiment.append(corpus.poa_confusion_matrix())
            moa_cm_per_experiment.append(corpus.moa_confusion_matrix())
            s = time.time() - t
            print("Fold took", s, "seconds")

            if i == 0:
                phoneme_type, phoneme_counts = np.unique(corpus.all_ref_phonemes, return_counts=True)
                phoneme_count_per_fold[fold - 1, :] = phoneme_counts
        per.append(per_per_experiment)
        poa_afer.append(poa_afer_per_experiment)
        moa_afer.append(moa_afer_per_experiment)
        poa_cm.append(poa_cm_per_experiment)
        moa_cm.append(moa_cm_per_experiment)

    with open('objs.pkl', 'wb') as file:
        pickle.dump([per, poa_afer, moa_afer, poa_cm, moa_cm, phoneme_count_per_fold], file)

with open('objs.pkl', 'rb') as file:
    per, poa_afer, moa_afer, poa_cm, moa_cm, phoneme_count_per_fold = pickle.load(file)

    experiment_folders = sorted(os.listdir("experiments"))
    experiment_renamer = {
        "baseline": "Baseline",
        "baseline_plus_oc": "Baseline + OC",
        "dnn_am_retraining": "DNN AM Retraining",
        "fhvae_input": "FHVAE",
        "fmllr": "fMLLR for AM Retraining"
    }
    experiments = [experiment_renamer[experiment] for experiment in experiment_folders]
    print(experiments)
    phoneme_barplot(per, phoneme_count_per_fold, experiments, "phoneme_2020_12_15")

    articulatory_barplot(moa_afer,experiments,
                         ["Vowel", "Plosive", "Nasal", "Fricative", "Affricates", "Approximant"],"moa_2020_12_15",
                         phoneme_count_per_fold,per)

    articulatory_barplot(poa_afer,experiments,
                         ["Vowel", "Bilabial", "Labiodental", "Dental",
                          "Alveolar", "Postalveolar", "Palatal", "Velar", "Glottal"],
                         "poa_2020_12_15",phoneme_count_per_fold,per)

    visualise_confusion_matrices(poa_cm,experiments,"confusion_matrix_poa_2020_12_15")

    visualise_confusion_matrices(moa_cm,experiments,"confusion_matrix_moa_2020_12_15")

