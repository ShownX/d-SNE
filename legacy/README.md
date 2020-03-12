# Introduction
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/d-sne-domain-adaptation-using-stochastic/domain-adaptation-on-svnh-to-mnist)](https://paperswithcode.com/sota/domain-adaptation-on-svnh-to-mnist?p=d-sne-domain-adaptation-using-stochastic)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/d-sne-domain-adaptation-using-stochastic/domain-adaptation-on-office-31)](https://paperswithcode.com/sota/domain-adaptation-on-office-31?p=d-sne-domain-adaptation-using-stochastic)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/d-sne-domain-adaptation-using-stochastic/domain-adaptation-on-visda2017)](https://paperswithcode.com/sota/domain-adaptation-on-visda2017?p=d-sne-domain-adaptation-using-stochastic)

MXNet/Gluon implementation for [d-SNE: Domain Adaptation using Stochastic Neighbourhood Embedding](https://arxiv.org/abs/1905.12775), which was accepted at CVPR 2019 as oral presentation.
 d-SNE aims to perform domain adaptation by aligning the source domain and target domain in a class by class fashion. d-SNE is a supervised learning algorithm and requires a few labeled samples from the target domain for training. The semi-supervised extension can further improve its performance by incoporating unlabeled target data. 

![Results](docs/tsne-digits.png)

# Prerequisites
## Dependencies
```bash
$ pip install -r requirements.txt

# Install the correct GPU version's mxnet
$ pip install mxnet-cu100 # for CUDA 10.0
```

## Datasets and abbreviations:
| Experiments | Datasets | Data Preparation Scripts|
| ----------- |:--------:|:--------:|
| Digits      | MNIST(MT), MNISTM(MM), SVHN(SN), and USPS(US)| |
| Office-31   | AMAZON (A), DSLR (D), and WEBCAM (W)         | |
| VisDA       | Synthetic (S) and Real (R)                   | |

Due to licensing issues, we won't be able to host the datasets but provide the scripts to prepare the datasets for each experiment.

## Download the pre-trained model
```bash
$ sh scripts/download_pretrained_models.sh
```
# Few Shot Domain Adaptation with d-SNE
## Model training
## 
## Digits Experiments: MT -> MM
Here, MNIST (MT) to MNISTM (MM) is used as an example 
For other experiments, just change the source and target abbr. names
- v0: Train on source domain and evaluate on the target domain
    ```bash
    $ python main_sda.py --method v0 --cfg cfg/digits-a.json --bb lenetplus --bs 256 --src MT --tgt MM --nc 10 --size 32 --train-src --log-itv 0 --dropout --hybridize
    ```
- v1: Train on source domain and random selected samples in the target domain
    ```bash
    $ python main_sda.py --method v1 --cfg cfg/digits-a.json --bb lenetplus --bs 256 --src MT --tgt MM --nc 10 --size 32 --train-src --log-itv 0 --dropout --hybridize
    ```
- v1: Fine-tune the model trained from the source selected samples in the target domain
    ```bash
    $ python main_sda.py --method v1 --cfg cfg/digits-a.json --bb lenetplus --bs 256 --src MT --tgt MM --nc 10 --size 32 --train-src --log-itv 0 --dropout --model-path path/to/model --hybridize
    ```
- ccsa \[1\]:
    ```bash
    $ python main_sda.py --method ccsa --cfg cfg/digits-a.json --bb lenetplus --bs 256 --src MT --tgt MM --nc 10 --size 32 --log-itv 0 --dropout --hybridize
    ```
- dsne: Train with dSNE
    ```bash
    $ python main_sda.py --method dsnet --cfg cfg/digits-a.json --bb lenetplus --bs 256 --src MT --tgt MM --nc 10 --size 32 --log-itv 100 --dropout --hybridize
    ```

- test:
    ```bash
    $ python main_sda.py --cfg cfg/digits-a.json --src US --tgt MT --method=dsnet --bb lenetplus --nc 10 --resize 32 --size 32 --dropout --postfix prefix/name --test --model-path model/path --plot
    ```

### Office Experiments: A -> D
#### Fine-tune the pre-trained model using the source domain

#### Adaptation with labeled target domain
- v0: Train on source domain and evaluate on the target domain
    ```bash
    $ python main_sda.py --method v0 --cfg cfg/offices.json --src A --tgt D --nc 31 --train-src --log-itv 0 --flip --random-color --random-crop --model-path path/to/model --fn --hybridize
    ```
- v1: Fine-tune with random selected samples in the target domain
    ```bash
    $ python main_sda.py --method v1 --cfg cfg/offices.json --src A --tgt D --nc 31 --log-itv 0 --flip --random-color --random-crop --model-path path/to/model --fn --hybridize
    ```
- dsnet: Train with dSNEt
    ```bash
    $ python main_sda.py --method dsnet --cfg cfg/offices.json --src A --tgt D --nc 31 --log-itv 100 --flip --random-color --random-crop --model-path path/to/model --fn --hybridize
    ```
### VisDA Experiments: S -> R
- v0: Train on source domain and evaluate on the target domain
    ```bash
    $ python main_sda.py --method v0 --cfg cfg/visa17.json --src S --tgt R --nc 12 --train-src --log-itv 0 --flip --random-color --random-crop --model-path path/to/model --fn --hybridize
    ```
- v1: Fine-tune with random selected samples in the target domain
    ```bash
    $ python main_sda.py --method v1 --cfg cfg/visa17.json --src S --tgt R --nc 12 --log-itv 0 --flip --random-color --random-crop --model-path path/to/model --fn --hybridize
    ```
- dsnet: Train with dSNEt
    ```bash
    $ python main_sda.py --method dsnet --cfg cfg/visa17.json --src S --tgt R --nc 12 --log-itv 100 --flip --random-color --random-crop --model-path path/to/model --fn --hybridize
    ```

# Semi-Supervised Domain Adaptation with d-SNE


# Reference
If you find any piece of this code or the paper useful, please cite our CVPR 2019 Oral paper:
```
@InProceedings{Xu_2019_CVPR,
    author = {Xu, Xiang and Zhou, Xiong and Venkatesan, Ragav and Swaminathan, Gurumurthy and Majumder, Orchid},
    title = {d-SNE: Domain Adaptation Using Stochastic Neighborhood Embedding},
    booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    month = {June 16-20},
    year = {2019},
    pages = {2497-2506}
}
```

