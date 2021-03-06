import torch
import string
import re
from collections import defaultdict


__table = str.maketrans('', '', string.punctuation)

def rouge_tokenize(text):
    '''A rough approximation of the original ROUGE perl script tokenizer.
       Removes punctuation, lowercases, and splits on whitespace.'''
    
    return re.split(r"\s+", text.translate(__table).lower(), flags=re.DOTALL)

def rouge_ngram_preprocess(document, summaries, ngram=1, length=100):

    vocab = {}

    summary_word_counts = []
    for summary in summaries:
        summary_word_count = defaultdict(int)
        summary_tokens = rouge_tokenize(summary)
        for token in summary_tokens[:length]:
            if token not in vocab:
                vocab[token] = len(vocab) + 2
            summary_word_count[vocab[token]] += 1
        summary_word_counts.append(summary_word_count)         
        
    summary_wc_tensor = torch.LongTensor(1, len(summaries), len(vocab) + 2)
    summary_wc_tensor.fill_(0)
    for i, wc in enumerate(summary_word_counts):
        for w, c in wc.items():
            summary_wc_tensor[0, i, w] = c
   
    doc_tensor = []
    for sentence in document:
        doc_tensor.append(
            [vocab.get(token, 1) for token in rouge_tokenize(sentence)])
    max_len = max([len(sent) for sent in doc_tensor])
    for sent in doc_tensor:
        if len(sent) < max_len:
            sent.extend([0] * (max_len - len(sent)))
    doc_tensor = torch.LongTensor(doc_tensor)

    return doc_tensor, summary_wc_tensor

def stack_documents(input):
    doc_size = max([input_i.size(0) for input_i in input])
    sent_size = max([input_i.size(1) for input_i in input])
    batch_size = len(input)
    output = torch.LongTensor(batch_size, doc_size, sent_size).fill_(0)
    for b, input_b in enumerate(input):
        output[b,:input_b.size(0),:input_b.size(1)].copy_(input_b)
    return output

def stack_word_counts(input):
    num_refs = max([input_i.size(1) for input_i in input])
    vocab_size = max([input_i.size(2) for input_i in input])
    batch_size = len(input) 
    output = torch.LongTensor(batch_size, num_refs, vocab_size).fill_(0)
    for b, input_b in enumerate(input):
        output[b,:input_b.size(1),:input_b.size(2)].copy_(input_b[0])
    return output
