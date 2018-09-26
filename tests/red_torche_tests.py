import unittest

import torch

import red_torche as rt
import rouge_papier


class TestRedTorche(unittest.TestCase):

    def assertTensorEqual(self, x, y, tolerance=1e-6):
        if isinstance(x, (torch.LongTensor)) or \
                isinstance(y, (torch.LongTensor)):
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
        rouge3 = rt.rouge_n(wc3, ref_wc3)

        self.assertTensorEqual(rouge3, expected_rouge3) 

    def test_mask_length(self):
       
        docs1 = torch.LongTensor(
            [[[2, 3, 1, 0],
              [5, 4, 1, 0],
              [2, 1, 5, 1],
              [2, 5, 4, 1]],
             [[1, 2, 3, 4],
              [4, 2, 0, 0],
              [1, 2, 3, 0],
              [5, 6, 7, 8]]])
        expected_masked_docs1 = torch.LongTensor(
            [[[2, 3, 1, 0],
              [5, 4, 1, 0],
              [2, 1, 5, 1],
              [0, 0, 0, 0]],
             [[1, 2, 3, 4],
              [4, 2, 0, 0],
              [1, 2, 3, 0],
              [5, 0, 0, 0]]])

        masked_docs1 = rt.mask_length(docs1, 10)
        self.assertTensorEqual(masked_docs1, expected_masked_docs1)
    
        docs2 = torch.LongTensor(
            [[2, 3, 0, 0],
             [5, 0, 0, 0],
             [2, 1, 5, 1],
             [2, 5, 4, 1],
             [3, 4, 5, 2]])

        expected_masked_docs2 = torch.LongTensor(
            [[[2, 3, 0, 0],
              [5, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0]]])

        masked_docs2 = rt.mask_length(docs2, 3)
        self.assertTensorEqual(masked_docs2, expected_masked_docs2)

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


    def test_rouge_recall_interface(self):

        docs = [["a b c d", "e f g h", "a b c e", "a a a b b"],
                ["d d d c e f", "z z d d", "e f f", "g f f e w", "a e f"]]
        summaries = [["a a a b c d e s a f g e a a s f a",
                      "z b b b d e a a g h t d k a j f s"],
                     ["z z d e f w d d   d a g h e k a s"]]

        docs_tensors = []
        ref_word_counts = []
        for docs_i, summaries_i in zip(docs, summaries):
            dt, wc = rt.rouge_ngram_preprocess(docs_i, summaries_i, length=10)
            docs_tensors.append(dt)
            ref_word_counts.append(wc)
        docs_tensors = rt.stack_documents(docs_tensors)  
        self.assertTrue(docs_tensors.size(0) == 2)
        self.assertTrue(docs_tensors.size(1) == 5)
        self.assertTrue(docs_tensors.size(2) == 6)
        ref_word_counts = rt.stack_word_counts(ref_word_counts)
        self.assertTrue(ref_word_counts.size(0) == 2)
        self.assertTrue(ref_word_counts.size(1) == 2)
        self.assertTrue(ref_word_counts.size(2) == 12)
        self.assertTrue(torch.all(ref_word_counts[0].sum(-1).eq(10)))
        self.assertTensorEqual(
            ref_word_counts[1].sum(-1).view(-1), torch.LongTensor([10, 0]))
        
        rouge = rt.rouge_recall(docs_tensors, ref_word_counts, length=10)
        ref_rouge = self.reference_rouge(
            ["\n".join(d) for d in docs], summaries, length=10)
        
        self.assertTensorEqual(rouge, ref_rouge)


    def test_rouge_recall_from_labels_interface(self):

        docs = [["a b c d", "e f g h", "a b c e", "a a a b b"],
                ["d d d c e f", "z z d d", "e f f", "g f f e w", "a e f"]]
        summaries = [["a a a b c d e s a f g e a a s f a",
                      "z b b b d e a a g h t d k a j f s"],
                     ["z z d e f w d d   d a g h e k a s"]]

        docs_tensors = []
        ref_word_counts = []
        for docs_i, summaries_i in zip(docs, summaries):
            dt, wc = rt.rouge_ngram_preprocess(docs_i, summaries_i, length=10)
            docs_tensors.append(dt)
            ref_word_counts.append(wc)
        docs_tensors = rt.stack_documents(docs_tensors)  
        self.assertTrue(docs_tensors.size(0) == 2)
        self.assertTrue(docs_tensors.size(1) == 5)
        self.assertTrue(docs_tensors.size(2) == 6)
        ref_word_counts = rt.stack_word_counts(ref_word_counts)
        self.assertTrue(ref_word_counts.size(0) == 2)
        self.assertTrue(ref_word_counts.size(1) == 2)
        self.assertTrue(ref_word_counts.size(2) == 12)
        self.assertTrue(torch.all(ref_word_counts[0].sum(-1).eq(10)))
        self.assertTensorEqual(
            ref_word_counts[1].sum(-1).view(-1), torch.LongTensor([10, 0]))
        
        batch_size = docs_tensors.size(0)
        doc_size = docs_tensors.size(1)
        
        labels1 = torch.distributions.Bernoulli(torch.tensor([.75])).sample(
            (batch_size, doc_size)).long().squeeze(-1)
        rouge1 = rt.rouge_recall_from_labels(
            docs_tensors, ref_word_counts, 
            labels1, length=10)

        sample_docs1 = []
        sample_refs1 = []
        for batch in range(batch_size):
                sample_doc = []
                for i, label in enumerate(
                        labels1[batch,:len(docs[batch])]):
                    if label.item() == 1:
                        sample_doc.append(docs[batch][i])
                sample_docs1.append("\n".join(sample_doc))
                sample_refs1.append(summaries[batch])
        ref_rouge1 = self.reference_rouge(
            sample_docs1, sample_refs1, length=10)
        self.assertTensorEqual(rouge1, ref_rouge1)
        
        sample_size = 3 
        
        labels2 = torch.distributions.Bernoulli(torch.tensor([.75])).sample(
            (batch_size, sample_size, doc_size)).long().squeeze(-1)
        
        rouge2 = rt.rouge_recall_from_labels(
            docs_tensors.unsqueeze(1), ref_word_counts.unsqueeze(1), 
            labels2, length=10)

        sample_docs2 = []
        sample_refs2 = []
        for batch in range(batch_size):
            for sample in range(sample_size):
                sample_doc = []
                for i, label in enumerate(
                        labels2[batch,sample,:len(docs[batch])]):
                    if label.item() == 1:
                        sample_doc.append(docs[batch][i])
                sample_docs2.append("\n".join(sample_doc))
                sample_refs2.append(summaries[batch])
        ref_rouge2 = self.reference_rouge(
            sample_docs2, sample_refs2, length=10).view(batch_size, 
                                                        sample_size)
        self.assertTensorEqual(rouge2, ref_rouge2)


    def test_rouge_recall_from_indices_interface(self):

        docs = [["a b c d", "e f g h", "a b c e", "a a a b b"],
                ["d d d c e f", "z z d d", "e f f", "g f f e w", "a e f"]]
        summaries = [["a a a b c d e s a f g e a a s f a",
                      "z b b b d e a a g h t d k a j f s"],
                     ["z z d e f w d d   d a g h e k a s"]]

        docs_tensors = []
        ref_word_counts = []
        for docs_i, summaries_i in zip(docs, summaries):
            dt, wc = rt.rouge_ngram_preprocess(docs_i, summaries_i, length=10)
            docs_tensors.append(dt)
            ref_word_counts.append(wc)
        docs_tensors = rt.stack_documents(docs_tensors)  
        self.assertTrue(docs_tensors.size(0) == 2)
        self.assertTrue(docs_tensors.size(1) == 5)
        self.assertTrue(docs_tensors.size(2) == 6)
        ref_word_counts = rt.stack_word_counts(ref_word_counts)
        self.assertTrue(ref_word_counts.size(0) == 2)
        self.assertTrue(ref_word_counts.size(1) == 2)
        self.assertTrue(ref_word_counts.size(2) == 12)
        self.assertTrue(torch.all(ref_word_counts[0].sum(-1).eq(10)))
        self.assertTensorEqual(
            ref_word_counts[1].sum(-1).view(-1), torch.LongTensor([10, 0]))
        
        batch_size = docs_tensors.size(0)
        doc_size = docs_tensors.size(1)
        
        indices1 = torch.LongTensor([[0,3,2],[1, 0, 2]])

        labels1 = torch.distributions.Bernoulli(torch.tensor([.75])).sample(
            (batch_size, doc_size)).long().squeeze(-1)
        rouge1 = rt.rouge_recall_from_indices(
            docs_tensors, ref_word_counts, 
            indices1, length=10)

        sample_docs1 = []
        sample_refs1 = []
        for batch in range(batch_size):
                sample_doc = []
                for idx in indices1[batch,:len(docs[batch])]:
                    sample_doc.append(docs[batch][idx])
                sample_docs1.append("\n".join(sample_doc))
                sample_refs1.append(summaries[batch])
        ref_rouge1 = self.reference_rouge(
            sample_docs1, sample_refs1, length=10)
        self.assertTensorEqual(rouge1, ref_rouge1)

        sample_size = 3 
        indices2 = torch.LongTensor([[[0, 3, 2], [1, 2, 3], [0, 3, 2]],  
                                     [[1, 0, 2], [4, 3, 2], [0, 4, 3]]])
        
        rouge2 = rt.rouge_recall_from_indices(
            docs_tensors, ref_word_counts.unsqueeze(1), 
            indices2, length=10)

        sample_docs2 = []
        sample_refs2 = []
        for batch in range(batch_size):
            for sample in range(sample_size):
                sample_doc = []
                for idx in indices2[batch,sample,:len(docs[batch])]:
                    sample_doc.append(docs[batch][idx])
                sample_docs2.append("\n".join(sample_doc))
                sample_refs2.append(summaries[batch])
        ref_rouge2 = self.reference_rouge(
            sample_docs2, sample_refs2, length=10).view(batch_size, 
                                                        sample_size)
        self.assertTensorEqual(rouge2, ref_rouge2)




def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestRedTorche("test_rouge_unigram_preprocess"))
    suite.addTest(TestRedTorche("test_word_counts"))
    suite.addTest(TestRedTorche("test_unigram_rouge1_recall_single_summary"))
    suite.addTest(TestRedTorche("test_mask_length"))
    suite.addTest(TestRedTorche("test_rouge_recall_interface"))
    suite.addTest(TestRedTorche("test_rouge_recall_from_labels_interface"))
    suite.addTest(TestRedTorche("test_rouge_recall_from_indices_interface"))
    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
