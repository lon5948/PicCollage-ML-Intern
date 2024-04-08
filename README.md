# PicCollage-ML-Intern-Call-Question

Predict the correlation between x and y values. To get acquainted with how it works, you can play the game here first: http://guessthecorrelation.com/

## Prepare

Downloading the [dataset](https://drive.google.com/file/d/1OqZJW8WUeUi2XrzJisJBHR4Er1lPowmG/view?usp=sharing)  which contains images of scatter plots and correlations corresponding to each scatter plot.

## Install

```bash
poetry install
```

## Run

```bash
cd src

python3 main.py --dir [data_directory] --batch_size [batch_size] --lr [learning rate] --epoch [epoch]
```

Remember, these parameters are not necessary, use `python3 main.py -h` to see the defaults.

## Results

|             | Epoch 1     |   Epoch 3 |    Epoch 5  |    Epoch 8  |   Epoch 10  |
| ----------- | ----------- | --------- | ----------- | ----------- | ----------- |
| Ground Truth|  -0.42736   | -0.42736  | -0.42736    | -0.42736    | -0.42736    |
| Prediction  | -0.04952    | -0.38341  | -0.41141    | -0.43451    | -0.42671    |
| Train Loss  | 0.0063360   | 0.0005271 | 0.0002327   | 0.0001397   | 0.0001226   |
| Test Loss   | 0.0007175   | 0.0003429 | 0.0002096   | 0.0001483   | 0.0001301   |
