
## Phoneme and articulatory analysis framework for Kaldi ASR

This work is the reproduction of the figures in the paper 'Low-Resource Automatic Speech Recognition and Error Analyses of Oral Cancer Speech'.

The code in this repository can be used to calculate PER and AFER based on a Kaldi wer details file.

## What does it do
* Implements a new variant of PER on word-level ASR
* Implements a new error rate, the AFER (Articulatory Feature Error Rate) on word-level ASR

The WER and the PER is calculated based on the Levensteihn distance.

### Requirements
- Use the conda environment.yml file provided
- sklearn version 0.25 will deprecate labels argument

### Paper reproduction
- Install the required packages for the framework
- Create the conda environment from environment yml
- Run analysis.py

### Adaptation for your own datasets
- Put your wer_details experiments in the same format as the other experiments, see example below
- Adapt the config files with your feature conversions and your lexicon, or use the already provided conversion
  tables or lexicons. (Provided for English and Dutch so far)
- There is an example provided below for PER extraction


### Example

```python
    from corpus import WERDetails
    from utils import HParam
    config = HParam("configs/dutch.yaml")
    wer_details = WERDetails("experiments/jasmin_example/scoring_kaldi/wer_details/per_utt", skip_calculation=False,
                             config=config)

    phoneme, other = wer_details.all_pers()
```

### Future plans
* Support for ESPNet WER details
* Support for multiple languages, and g2p models
