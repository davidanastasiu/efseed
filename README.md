## EF-SEED

We present `EF-SEED`,  Enhanced Feature-based SEED (EF-SEED)  to considerably enhance streamflow prediction using multivariate time series data from rain and streamflow sensors. This work is expanded on the prior paper "Seed: An effective model for highly-skewed streamflow time series data forecasting". Please note that, in the IEEE Access article below, the method is known as `MSEED`. If you make use of our code or data, please cite either or both our papers below.


```bibtex
@article{li2024Mseed,
  title={Multivariate Segment Expandable Encoder-Decoder Model for Time Series Forecasting},
  author={Li, Yanhong and Anastasiu, David C},
  journal={IEEE Access},
  year={2024},
  publisher={IEEE}
}

@inproceedings{li2023seed,
  title={Seed: An effective model for highly-skewed streamflow time series data forecasting},
  author={Li, Yanhong and Xu, Jack and Anastasiu, David C},
  booktitle={2023 IEEE International Conference on Big Data (BigData)},
  pages={728--737},
  year={2023},
  organization={IEEE}
}
```

## Preliminaries

Experiments were executed in an Anaconda 3 environment with Python 3.8.8. The following will create an Anaconda environment and install the requisite packages for the project.

```bash
conda create -n EFSEED python=3.8.8
conda activate EFSEED
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 -c pytorch
python -m pip install -r requirements.txt
```

## Files organizations

Download the datasets from [here](https://clp.engr.scu.edu/static/datasets/seed_datasets.zip) and upzip the files in the data_provider directory. In the ./data_provider/datasets directory, there should now be 4 stream sensor (file names end with _S_fixed.csv) and 4 rain sensor (file names end with _R_fixed.csv) datasets.


## Parameters setting

--stream_sensor: stream dataset file name. The file should be csv file.

--rain_sensor: rain dataset file name. The file should be csv file.

--train_volume: train set size.

--hidden_dim: hidden dim of lstm layers.

--cnn_dim: hidden dim of cnn layers.

--layer: number of layers.

--os_h: oversampling steps.

--os_l: oversampling frequency. 

--r_shift: the length of rain data ahead to faciliate forecasting. 60 suggested in these datasets.

--watershed: 1 if trained with rain info, else 0.

--quantile: quantile value, set according to the data distribution.

--model: model name, used to generate the pt file and predicted file names.

--mode: set it to 'train' or 'inference' with an existing pt_file.

--pt_file: if set, the model will be loaded from this pt file, otherwise check the file according to the assigned parameters.

--save: if save the predicted file of testset, set to 1, else 0.

--outf: default value is './output', the model will be saved in the train folder in this directory.

Refer to the annotations in `run.py` for other parameter settings. Default parameters for reproducing are set in the files (file names start with opt and end with .txt) under './models/'.

## Training and Inferencing

Execute the Jupyter notebook experiments.ipynb to train models and conduct inferences on the test sets of the four stream datasets described in the associated paper.

The Jupyter notebook example.ipynb shows how to train a model via command line commands and use specific model functions to perform inference on the SFC sensor dataset.
