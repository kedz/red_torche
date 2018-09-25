import red_torche as rt
import torch


def test_rouge_ngram_preprocess():

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
        [[0, 0, 1, 2, 2, 1, 0],
         [0, 0, 3, 1, 1, 0, 1]])

    print(summaries_tensor)
    print(expected_summaries_tensor) 
    print(document_tensor)
    print(expected_document_tensor)

test_rouge_ngram_preprocess()


def test_rouge1_recall_single_summary():
    pass

def test_word_counts():
    summary_sents = ["a b c", "d e f", "a c d z a"]
    ref_summaries = ["a a a b e d d d d e"]
    summary_tensor, ref_wc = rt.rouge_ngram_preprocess(
        summary_sents, ref_summaries, length=10)
    wc1 = rt.word_counts(summary_tensor, ref_wc.size(1))
                                         #  a  b  e  d
    expected_wc1 = torch.LongTensor([[0, 4, 3, 1, 1, 2,]])

    print(wc1)
    print(expected_wc1)

print("\n\n\n")
test_word_counts()


