import argparse
from dsne.utils import get_packed_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='', help='dataset directory')
    parser.add_argument('--dataset', type=str, default='mnist', help='dataset name')
    parser.add_argument('--dump_path', type=str, default='', help='dataset name')
    args = parser.parse_args()

    dataset = get_packed_dataset(args.dataset)(args.root)
    dataset.read()
    dataset.dump(args.dump_path)

