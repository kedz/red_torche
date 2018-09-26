import torch

from functools import reduce
import operator 

from .rouge_n_impl import mask_length, word_counts, rouge_n


def rouge_recall(predicted_summaries, reference_word_counts, length=100):
    
    assert predicted_summaries.size(0) == reference_word_counts.size(0)

    predicted_summaries = mask_length(predicted_summaries, length)

    max_vocab_size = reference_word_counts.size(-1)
    predicted_word_counts = word_counts(predicted_summaries, max_vocab_size)
   
    rouge = rouge_n(predicted_word_counts, reference_word_counts)
    
    return rouge 

def rouge_recall_from_labels(documents, reference_word_counts, labels, 
                             length=100):

    predicted_summaries = mask_length(
        labels.unsqueeze(-1) * documents, length)

    max_vocab_size = reference_word_counts.size(-1)
    predicted_word_counts = word_counts(predicted_summaries, max_vocab_size)
   
    rouge = rouge_n(predicted_word_counts, reference_word_counts)
    
    return rouge 

def rouge_recall_from_indices(documents, reference_word_counts, indices,
                              length=100):

    batch_indices = torch.arange(
        indices.size(0), out=documents.new()).view(-1, 1)

    batch_indices = torch.arange(
        indices.size(0), out=documents.new()).view(-1, 1).repeat(
            1, reduce(operator.mul, indices.size()[1:], 1)).view(-1)
    
    predicted_summaries = documents[batch_indices, indices.view(-1)].view(
        *indices.size(), -1)
    
    predicted_summaries = mask_length(predicted_summaries, length)

    max_vocab_size = reference_word_counts.size(-1)
    predicted_word_counts = word_counts(predicted_summaries, max_vocab_size)
   
    rouge = rouge_n(predicted_word_counts, reference_word_counts)
 
    return rouge
