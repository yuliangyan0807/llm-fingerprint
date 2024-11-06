from datasets import load_from_disk

if __name__ == '__main__':
    data = load_from_disk('./data/trajectory_set')
    print(len(data))
    