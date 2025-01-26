from streaming import MDSWriter
from tqdm import tqdm


import os
import random
import datasets


def filter_text_field(example, text_field='text'):
    return {'text': example[text_field]}

def convert_hf_to_mds():

    train_path = os.path.join('/Volumes/TrainData/cleaned-kr-data', 'train')
    # train_path = os.path.join('./cleaned-kr-data', 'train')


    dataset = datasets.load_dataset('HuggingFaceFW/fineweb-2', 'kor_Hang')
    # train_data = dataset.split('train')

    # for example in train_data:
    #     break

    # with MDSWriter(
    #     columns = {"text": "str"}, out=train_path, compression=None
    # ) as out:
        

    # train_samples = samples[:split_idx]
    # val_samples = samples[split_idx:]

    # with MDSWriter(
    #     columns = {"text": "str"}, out=os.path.join('ModernBERT/work/mds', 'train'), compression=None
    # ) as out:
    #     for sample in tqdm(train_samples, desc='train'):
    #         out.write(sample)

    # with MDSWriter(
    #     columns = {"text": "str"}, out=os.path.join('ModernBERT/work/mds', 'val'), compression=None
    # ) as out:
    #     for sample in tqdm(val_samples, desc='val'):
    #         out.write(sample)


if __name__ == "__main__":
    convert_hf_to_mds()