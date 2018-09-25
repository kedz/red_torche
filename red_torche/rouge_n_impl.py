import torch
from functools import reduce
import operator


def word_counts(input, vocab_size):
    '''Converts tensor of dimension ... x sent x word into a ... x vocab_size
       word count tensor.'''

    assert input.dim() >= 2

    batch_dims = input.size()[:-2]
    if not batch_dims:
        batch_dims = (1,)
        flat_batch_size = 1
    else:
        flat_batch_size = reduce(operator.mul, batch_dims, 1)

    doc_size = input.size(-2)
    sent_size = input.size(-1)

    wc_col_indices = input.view(-1)
    wc_values = wc_col_indices.gt(0).long()

    # Create row index data for sparse matrix. E.g. 0, 0, 0, ..., 1, 1, 1, ...
    wc_row_indices = torch.arange(flat_batch_size, out=input.new()).view(
        -1, 1).repeat(1, doc_size * sent_size).view(-1)

    sparse_wc = torch.sparse.LongTensor(
        torch.stack([wc_row_indices, wc_col_indices]).cpu(),
        wc_values.cpu(),
        (flat_batch_size, vocab_size))
    
    if input.is_cuda:
        sparse_wc = sparse_wc.cuda(input.device)

    return sparse_wc.coalesce().to_dense().view(*batch_dims, -1)    

def rouge_n(predicted_word_counts, reference_word_counts, reduction="mean"):
    predicted_word_counts = predicted_word_counts.unsqueeze(-2)
    overlap = torch.min(
        predicted_word_counts, reference_word_counts)

    total_matches = overlap.sum(-1).sum(-1).float()
    total_ref_ngrams = reference_word_counts.sum(-1).sum(-1).float()
    rouge = total_matches / total_ref_ngrams 
    return rouge
    #if reduction == "mean" and rouge.size(-1) > 1:
    #    return rouge.mean(-1, keepdim=True)
    #else:
    #    return rouge

def mask_length(inputs, length):
    pass
