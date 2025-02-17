from transformers import pipeline
import os
import chardet
import ftfy


def test():

    ner = pipeline(
        task="ner",
        model="./results/checkpoint-1000",
    )

    test_sentence = "영남지방에서 가장 큰 나무는 어디에 있는 것일까?"

    for x in ner(test_sentence):
        # str
        print(x['word'])
        print(x['word'].encode('utf-8'))
        # print(bytes(x['word'], 'utf-8'))


if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    test()