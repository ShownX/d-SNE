## How to process the dataset
### Digit datasets
Download the digits from the original sites. Due to the license, this repository does not provide any format of data.

For digits, we provide the following sample code to pack the data used in this repository, assuming that the dataset is stored under the folder ```datasets```.

- MNIST
    ```shell script
    $ python tools/pack_dataset.py --root datasets/MNIST --dataset mnist --dump_path datasets/MNIST.bin
    ```

- MNIST-M
    ```shell script
    $ python tools/pack_dataset.py --root datasets/MNIST-M --dataset mnistm --dump_path datasets/MNIST-M.bin
    ```

### Office31 dataset


### VisDA dataset
