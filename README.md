# ELMoConverter
Convert ELMoForManyLangs ELMo file into AllenNLP format

# Instalattion
## Install ELMoForManyLangs
See [ELMoForManyLangs](https://github.com/HIT-SCIR/ELMoForManyLangs/tree/master)  

```sh
$ git clone https://github.com/HIT-SCIR/ELMoForManyLangs.git
$ cd ELMoForManyLangs
$ python setup.py install
```

## Install library
```sh
$ pip install -r requirements.tx
```

## Usage
```
usage: convert.py [-h] [--model_path MODEL_PATH] [--output_path OUTPUT_PATH]

optional arguments:
  -h, --help                      show this help message and exit
  --model_path MODEL_PATH         Path to elmo trained with ELMoForManyLangs
  --output_path OUTPUT_PATH       Output path of elmo for allennlp
```
