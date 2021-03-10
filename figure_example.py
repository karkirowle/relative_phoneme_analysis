
from corpus import WERDetails
from utils import HParam
#partition = "test"
#number_of_phonemes = 40
preprocessing = True

import numpy as np
import pandas as pd

if preprocessing:
    config = HParam("configs/dutch.yaml")
    wer_details = WERDetails("experiments/jasmin_example/scoring_kaldi/wer_details/per_utt", skip_calculation=False,
                             config=config)

    phoneme, other = wer_details.all_pers()

    print(len(phoneme))
    print(len(other))
    df = pd.DataFrame(data=other, index=phoneme)
    #df = pd.DataFrame(data=np.vstack((phoneme,other)))
    print(df)
    #print(phoneme)
    #print(other)