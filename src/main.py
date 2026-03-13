from __future__ import annotations

from dataset import DatasetFactory, DatasetConfig, DatasetType
from parser import get_args_parser, get_config


def main(config) -> int:


    dataset_config = DatasetConfig(type=DatasetType.IMAGENET)
    dataset = DatasetFactory.create(dataset_config)

    print(len(dataset))
    print(dataset_config.get_data_path())

    return 0


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    config = get_config(args)

    main(config)
