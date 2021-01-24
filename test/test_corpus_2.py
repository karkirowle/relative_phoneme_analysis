from unittest import TestCase

from corpus_2 import WERDetails


class TestWERDetails(TestCase):
    def test_poa_to_phonemes(self):
        wer_details = WERDetails("../experiments/baseline/wer_details/per_utt", skip_calculation=True)

        self.assertEqual(sorted(wer_details.poa_to_phonemes("Bilabial")), sorted(["B", "P","EM","W","M"]))

    def test_phoneme_to_moa(self):
        wer_details = WERDetails("../experiments/baseline/wer_details/per_utt", skip_calculation=True)
        self.assertEqual(wer_details.phoneme_to_moa("P"), "Plosive")

    def test_phoneme_to_poa(self):
        wer_details = WERDetails("../experiments/baseline/wer_details/per_utt", skip_calculation=True)
        self.assertEqual(wer_details.phoneme_to_poa("P"), "Bilabial")

    def test_word_in_cmu_dict(self):
        wer_details = WERDetails("../experiments/baseline/wer_details/per_utt", skip_calculation=True)
        self.assertEqual(wer_details.word_in_cmu_dict("{asdfhj}"), True)
        self.assertEqual(wer_details.word_in_cmu_dict("asdfhj"), False)

    def test_word_to_phoneme(self):
        wer_details = WERDetails("../experiments/baseline/wer_details/per_utt", skip_calculation=True)
        self.assertEqual(wer_details.word_to_phoneme("cheese",True), ["CH", "IY", "Z"])
        self.assertEqual(wer_details.word_to_phoneme("cheese",False), ["CH", "IY1", "Z"])

    def test_arpabet_cleaner(self):
        self.assertEqual(WERDetails.arpabet_cleaner("{CH} {IY1} {Z}", True),["CH","IY","Z"])
        self.assertEqual(WERDetails.arpabet_cleaner("{CH} {IY1} {Z}", False), ["CH", "IY1", "Z"])

    def test_clean_non_words(self):
        self.assertEqual(WERDetails.clean_non_words(["aaa","***","bbb"]),["aaa","bbb"])