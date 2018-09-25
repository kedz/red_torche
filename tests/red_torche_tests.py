import unittest

import torch

import red_torche as rt
import rouge_papier


class TestRedTorche(unittest.TestCase):

    def assertTensorEqual(self, x, y, tolerance=1e-6):
        if isinstance(x, (torch.LongTensor, torch.cuda.LongTensor)) or \
                isinstance(y, (torch.LongTensor, torch.cuda.LongTensor)):
            self.assertTrue(torch.all(torch.abs(x - y).float().lt(tolerance)))
        else:
    
            diff = x - y
            inf_mask = x.eq(float("inf"))
            ninf_mask = x.eq(float("-inf"))
            self.assertTrue(torch.equal(inf_mask, y.eq(float("inf"))))
            self.assertTrue(torch.equal(ninf_mask, y.eq(float("-inf"))))
    
            diff = x - y
            diff[inf_mask] = 0
            diff[ninf_mask] = 0
    
            self.assertTrue(torch.all(torch.abs(diff).lt(tolerance)))


    def test_rouge_unigram_preprocess(self):
    
        document = ["a b c d.", "b e f g.", "a b e f z."]  
        summaries = ["a c.\n d g?\n c d. \n q e d.",
                     "a, a. ``a c q d.''"]
        document_tensor, summaries_tensor = rt.rouge_ngram_preprocess(
            document, summaries, length=6) 
    
        expected_document_tensor = torch.LongTensor(
            [[2, 1, 3, 4, 0],
             [1, 1, 1, 5, 0],
             [2, 1, 1, 1, 1]])
        expected_summaries_tensor = torch.LongTensor(
            [[[0, 0, 1, 2, 2, 1, 0],
              [0, 0, 3, 1, 1, 0, 1]]])
    
        self.assertTensorEqual(document_tensor, expected_document_tensor)
        self.assertTensorEqual(summaries_tensor, expected_summaries_tensor)

    def test_word_counts(self):
        summary_sents = ["a b c", "d e f", "a c d z a"]
        ref_summaries = ["a a a b e d d d d e"]
        summary_tensor, ref_wc = rt.rouge_ngram_preprocess(
            summary_sents, ref_summaries, length=10)
        wc1 = rt.word_counts(summary_tensor, ref_wc.size(-1))
                                             #  a  b  e  d
        expected_wc1 = torch.LongTensor([[0, 4, 3, 1, 1, 2,]])
    
        self.assertTensorEqual(wc1, expected_wc1)

    def test_unigram_rouge1_recall_single_summary(self):
        summary_sents1 = ["a b c", "d e f", "a c d z"]
        ref_summaries1 = ["a a a b e d d d d e"]
        summary_tensor1, ref_wc1 = rt.rouge_ngram_preprocess(
            summary_sents1, ref_summaries1, length=10)
        wc1 = rt.word_counts(summary_tensor1, ref_wc1.size(-1))
        expected_rouge1 = self.reference_rouge(
            ["\n".join(summary_sents1)], [ref_summaries1])
        rouge1 = rt.rouge_n(wc1, ref_wc1)
        self.assertTensorEqual(rouge1, expected_rouge1)
        ref_summaries2 = ["a ``b'' b b e d d d d a f"]
        summary_tensor2, ref_wc2 = rt.rouge_ngram_preprocess(
            summary_sents1, ref_summaries2, length=10)
        wc2 = rt.word_counts(summary_tensor2, ref_wc2.size(-1))
        expected_rouge2 = self.reference_rouge(
            ["\n".join(summary_sents1)], [ref_summaries2], length=10)
        rouge2 = rt.rouge_n(wc2, ref_wc2)

        self.assertTensorEqual(rouge2, expected_rouge2)

        ref_summaries3 = ["a a a b e d d d d e", 
                          "a g g g e d d d d a f"]
        summary_tensor3, ref_wc3 = rt.rouge_ngram_preprocess(
            summary_sents1, ref_summaries3, length=10)
        wc3 = rt.word_counts(summary_tensor3, ref_wc3.size(-1))
        expected_rouge3 = self.reference_rouge(
            ["\n".join(summary_sents1)], [ref_summaries3], length=10)
        rouge3 = rt.rouge_n(wc3, ref_wc3, reduction=None)

        self.assertTensorEqual(rouge3, expected_rouge3) 



    def test_mask_length(self):
        pass


    def reference_rouge(self, system_summaries, reference_summaries, order=1,
                        length=100):
        with rouge_papier.util.TempFileManager() as manager:
            path_data = []
            for system_sum, ref_sums in zip(system_summaries, 
                                            reference_summaries):    
                sys_sum_path = manager.create_temp_file(system_sum)
                ref_sum_paths = [manager.create_temp_file(ref_sum)
                                 for ref_sum in ref_sums]
                path_data.append([sys_sum_path, ref_sum_paths])
            config_text = rouge_papier.util.make_simple_config_text(path_data)
            config_path = manager.create_temp_file(config_text)
            df = rouge_papier.compute_rouge(
                config_path, max_ngram=order, lcs=False, 
                remove_stopwords=False,
                length=length)
            return torch.FloatTensor(df.values[:-1, order - 1])

def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestRedTorche("test_rouge_unigram_preprocess"))
    suite.addTest(TestRedTorche("test_word_counts"))
    suite.addTest(TestRedTorche("test_unigram_rouge1_recall_single_summary"))
    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())



exit()

print("\n\n\n")
test_word_counts()


