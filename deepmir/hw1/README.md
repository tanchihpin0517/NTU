## Environment
Python version: 3.10

Install dependencies with `pip install -r requirements.txt
`

## Prepare
Put all the data into `data` directory like this:
```
data
├── artist20
└── artist20_testing_data
```
Use modified `split_artist20.py` to generate training and validation song set in the format of tsv file:
```
python -m hw1.split_artist20
```
and run `prepare_training_data.sh` and `prepare_testing_data.sh`

(note: `prepare_training_data.sh` is required even you just want to do prediction)

## Training
```
./train.sh --batch_size [BATCH_SIZE]
```

## Predict
Download the checkpoint from this [link](https://www.dropbox.com/scl/fi/gkgzrvntww5rc5jtcp4h2/epoch-028-tl-0.00-v1-0.66-v3-0.84-step-30914.ckpt?rlkey=vdwerwwlgok4t94q8v501iw41&dl=0), and run:
```
./test.sh --ckpt_file [CHECKPOINT_FILE]
```
