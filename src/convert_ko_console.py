from streaming import StreamingDataset

import os
import json

def main():
    # 원격 및 로컬 경로 설정
    remote = None
    local = './my-copy-c4/ko/train_small'

    # StreamingDataset 생성
    datasets = StreamingDataset(local=local, remote=remote, shuffle=False, batch_size=100)

    # my-copy-c4/tmp/train_small.txt로 텍스트 파일로 저장 
    disk_size = 0
    for dataset in datasets:
        disk_size += len(json.dumps(dataset))

    print(f"Disk size: {disk_size/1024/1024} MBytes")

if __name__ == "__main__":
    main()