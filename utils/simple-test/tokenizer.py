from tokenizers import BertWordPieceTokenizer

def tokenize():
    bert_wordpiece_tokenizer = BertWordPieceTokenizer()
    bert_wordpiece_tokenizer.train(
        files=["./utils/simple-test/small.txt"],
        vocab_size=10,
        min_frequency=1,
        limit_alphabet=1000,
        initial_alphabet=[],
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        show_progress=True,
        wordpieces_prefix="##",
    )

    vocab = bert_wordpiece_tokenizer.get_vocab() 
    sorted(vocab, key=lambda x: vocab[x])

    encoding = bert_wordpiece_tokenizer.encode("ABCDE")
    print(encoding.tokens)
    print(encoding.ids)

    

if __name__ == "__main__":
    tokenize()