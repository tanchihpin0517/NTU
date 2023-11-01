## Environment
Python version: 3.10

Install dependencies with `pip install -r requirements.txt
`

## Prepare
Put all the data into `data` directory like this:
```
data
├── m4singer
├── m4singer_valid
├── m4singer_valid.zip
├── m4singer.zip
├── testing_mel
└── testing_mel.zip
```
Run:
```
python -m hw2.resample && ./prepare_valid_data.sh
```
to generate required data for training and testing.
The directory structure should be like this once the script is done:
```
data
├── m4singer
├── m4singer_22050
├── m4singer_valid
├── m4singer_valid_22050
├── m4singer_valid_22050_mel
├── m4singer_valid.zip
├── m4singer.zip
├── testing_mel
└── testing_mel.zip
```

## Training
```
train_hifigan_hw2_spec.sh
```

## Predict
Download the checkpoint from this [link](https://www.dropbox.com/scl/fi/b0kdwrxufq91tp5jm8b6j/g_00050000?rlkey=20bved2g0ijev9zx0yycnlv3u&dl=0) to `./cp_hifigan_hw2_spec`, and run:
```
test_hifigan.sh
```
