from streaming import StreamingDataset

datasets = StreamingDataset(local=local, remote=remote, shuffle=False, batch_size=100)

