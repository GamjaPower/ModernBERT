from streaming import MDSWriter
from tqdm import tqdm

import os
import random

def convert_html_to_txt(input_dir, output_dir):
    """HTML 파일을 txt 파일로 변환하는 함수"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

   
    samples = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            input_file_path = os.path.join(root, file)
            if not file.startswith('b'):
                continue

            with open(input_file_path, 'r') as infile:
                html_content = infile.read()
                samples.append({"text": html_content})

    # Shuffle samples
    random.shuffle(samples)

    # Split samples into train and val
    split_idx = int(len(samples) * 0.9)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    with MDSWriter(
        columns = {"text": "str"}, out=os.path.join('ModernBERT/work/mds', 'train'), compression=None
    ) as out:
        for sample in tqdm(train_samples, desc='train'):
            out.write(sample)

    with MDSWriter(
        columns = {"text": "str"}, out=os.path.join('ModernBERT/work/mds', 'val'), compression=None
    ) as out:
        for sample in tqdm(val_samples, desc='val'):
            out.write(sample)


if __name__ == "__main__":
    input_directory = './work/txt'
    output_directory = './work/mds'
    convert_html_to_txt(input_directory, output_directory)