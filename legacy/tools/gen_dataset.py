import argparse
from dsne.utils.enroll_dataset import get_packed_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='', help='dataset directory')
    parser.add_argument('--dataset', type=str, default='mnist', help='dataset name')

    args = parser.parse_args()

    dataset = get_packed_dataset(args.dataset)(args.dir)
    dataset.read()
    dataset.dump()
