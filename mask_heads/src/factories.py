from typing import Optional

from src import datasets, metrics, models
from src.configs import DataConfigs, DecoderConfigs, ModelConfigs


def get_dataset(data_configs: DataConfigs, **kwargs):
    return getattr(datasets, data_configs.name)(
        data_configs=data_configs,
        **kwargs,
    )


def get_model(
    model_configs: ModelConfigs,
    decoder_configs: DecoderConfigs,
):
    return getattr(models, decoder_configs.method)(
        model_configs=model_configs,
        decoder_configs=decoder_configs,
    )


def get_metrics(data_configs: DataConfigs):
    return getattr(metrics, data_configs.name)()
