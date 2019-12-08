import argparse
import json
import os
from pathlib import Path

from ELMoForManyLangs.elmoformanylangs import Embedder
from util import convert_config, create_elmo_h5_from_embedder


def convert(args):
    model_path = args.model_path
    output_path = args.output_path
    config_path = Path(model_path, 'configs')
    print(config_path)
    config_file = os.listdir(config_path)[0]
    if not config_file.endswith('.json'):
        raise ValueError('Only single config file should be put.')

    with (config_path / config_file).open() as f:
        config = json.load(f)

    # Convert config json
    allennlp_config = convert_config(config)
    config_output_name = 'allennlp_config.json'
    with Path(output_path, config_output_name).open(mode='w') as f:
        json.dump(allennlp_config, f, indent=2)

    # Convert ELMo
    embedder = Embedder(model_path)
    model_output_name = 'allennlp_elmo.hdf5'
    output_file = os.path.join(output_path, model_output_name)
    create_elmo_h5_from_embedder(embedder, output_file, config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", help="Path to elmo trained with ELMoForManyLangs")
    parser.add_argument("--output_path", help="Output path of elmo for allennlp")
    args = parser.parse_args()
    convert(args)
