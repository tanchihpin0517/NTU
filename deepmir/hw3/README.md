## Environment
Python version: 3.10

Install dependencies with `pip install -r requirements.txt
`

## Prepare
Extract `dataset` into `data` directory like this:
```
data
└── dataset
```
Run:
```
./gen_vocab.sh
./gen_data_cache.sh
```
to generate required data for training.
The directory structure should be like this once the script is done:
```
data
├── dataset
├── dataset_cache_bd_4
├── dataset_cache_full_bd_4.pkl
└── vocab.json
```

## Training
```
./train_v1.sh # or train_v2.sh
```

## Predict
Download the checkpoint from this [link](https://www.dropbox.com/scl/fi/k26tsdzdgpq7jjcb3sevs/model_00150000?rlkey=fn3v0ypdlkwrsy961ke3c44ug&dl=0) to `./ckpt/v1`, and run:
```
mkdir ./output
./inference.sh v1 00150000
```
