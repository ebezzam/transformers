# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np

from ...configuration_utils import PretrainedConfig


class VibeVoiceAcousticTokenizerEncoderConfig(PretrainedConfig):
    r"""
    Configuration class for [`VibeVoiceAcousticTokenizerEncoderModel`].

    Args:
        channels (`int`, *optional*, defaults to 1):
            Number of input channels.
        hidden_size (`int`, *optional*, defaults to 64):
            Dimensionality of latent representations.
        kernel_size (`int`, *optional*, defaults to 7):
            Kernel size for convolutional layers.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            Epsilon value for RMSNorm layers.
        layer_scale_init_value (`float`, *optional*, defaults to 1e-06):
            Initial value for layer scaling.
        initializer_range (`float`, *optional*, defaults to 0.01):
            Standard deviation for weight initialization.
        num_filters (`int`, *optional*, defaults to 32):
            Number of filters in initial convolutional layer, doubles after each downsampling.
        downsampling_ratios (`List[int]`, *optional*, defaults to `[2, 2, 4, 5, 5, 8]`):
            Downsampling ratios for each layer.
        depths (`List[int]`, *optional*, defaults to `[3, 3, 3, 3, 3, 3, 8]`):
            Number of ConvNeXt blocks at each stage.
        hidden_act (`str`, *optional*, defaults to `"gelu"`):
            Activation function to use.
        ffn_expansion (`int`, *optional*, defaults to 4):
            Expansion factor for feed-forward networks.
    """

    model_type = "vibevoice_acoustic_tokenizer_encoder"

    def __init__(
        self,
        channels=1,
        hidden_size=64,
        kernel_size=7,
        rms_norm_eps=1e-5,
        layer_scale_init_value=1e-6,
        initializer_range=1e-2,
        num_filters=32,
        downsampling_ratios=[2, 2, 4, 5, 5, 8],
        depths=[3, 3, 3, 3, 3, 3, 8],
        hidden_act="gelu",
        ffn_expansion=4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.channels = channels
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.kernel_size = kernel_size
        self.rms_norm_eps = rms_norm_eps
        self.layer_scale_init_value = layer_scale_init_value
        self.ffn_expansion = ffn_expansion
        self.initializer_range = initializer_range
        self.num_filters = num_filters
        self.downsampling_ratios = downsampling_ratios
        self.depths = depths

    @property
    def hop_length(self):
        return np.prod(self.downsampling_ratios)


class VibeVoiceAcousticTokenizerDecoderConfig(PretrainedConfig):
    r"""
    Configuration class for [`VibeVoiceAcousticTokenizerDecoderModel`].

    Args:
        channels (`int`, *optional*, defaults to 1):
            Number of output channels.
        hidden_size (`int`, *optional*, defaults to 64):
            Dimensionality of latent representations.
        kernel_size (`int`, *optional*, defaults to 7):
            Kernel size for convolutional layers.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            Epsilon value for RMSNorm layers.
        layer_scale_init_value (`float`, *optional*, defaults to 1e-06):
            Initial value for layer scaling.
        initializer_range (`float`, *optional*, defaults to 0.01):
            Standard deviation for weight initialization.
        num_filters (`int`, *optional*, defaults to 32):
            Number of filters after final upsampling.
        upsampling_ratios (`List[int]`, *optional*, defaults to `[8, 5, 5, 4, 2, 2]`):
            Upsampling ratios for each layer (reverse of encoder downsampling).
        decoder_depths (`List[int]`, *optional*, defaults to `[8, 3, 3, 3, 3, 3, 3]`):
            Number of ConvNeXt blocks at each decoder stage (reverse of encoder depths).
        hidden_act (`str`, *optional*, defaults to `"gelu"`):
            Activation function to use.
        ffn_expansion (`int`, *optional*, defaults to 4):
            Expansion factor for feed-forward networks.
    """

    model_type = "vibevoice_acoustic_tokenizer_decoder"

    def __init__(
        self,
        channels=1,
        hidden_size=64,
        kernel_size=7,
        rms_norm_eps=1e-5,
        layer_scale_init_value=1e-6,
        initializer_range=1e-2,
        num_filters=32,
        upsampling_ratios=[8, 5, 5, 4, 2, 2],
        decoder_depths=[8, 3, 3, 3, 3, 3, 3],
        hidden_act="gelu",
        ffn_expansion=4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.channels = channels
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.kernel_size = kernel_size
        self.rms_norm_eps = rms_norm_eps
        self.layer_scale_init_value = layer_scale_init_value
        self.ffn_expansion = ffn_expansion
        self.initializer_range = initializer_range
        self.num_filters = num_filters
        self.upsampling_ratios = upsampling_ratios
        self.decoder_depths = decoder_depths


class VibeVoiceAcousticTokenizerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`VibeVoiceAcousticTokenizerModel`]. It is used to
    instantiate a VibeVoice acoustic tokenizer model according to the specified arguments, defining the model
    architecture. Instantiating a configuration with the defaults will yield a similar configuration of the acoustic
    tokenizer within the VibeVoice architecture.

    e.g. [microsoft/VibeVoice-1.5B](https://huggingface.co/microsoft/VibeVoice-1.5B)

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        encoder_config (`VibeVoiceAcousticTokenizerEncoderConfig` or `dict`, *optional*):
            Configuration for the encoder. If not provided, will be created from the flat parameters.
        decoder_config (`VibeVoiceAcousticTokenizerDecoderConfig` or `dict`, *optional*):
            Configuration for the decoder. If not provided, will be created from the flat parameters.
        vae_std (`float`, *optional*, defaults to 0.625):
            Standard deviation used for VAE sampling after encoder.
        channels (`int`, *optional*, defaults to 1):
            Number of input/output channels. Used for backward compatibility when encoder_config/decoder_config not provided.
        hidden_size (`int`, *optional*, defaults to 64):
            Dimensionality of latent representations. Used for backward compatibility.
        kernel_size (`int`, *optional*, defaults to 7):
            Kernel size for convolutional layers. Used for backward compatibility.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            Epsilon value for RMSNorm layers. Used for backward compatibility.
        layer_scale_init_value (`float`, *optional*, defaults to 1e-06):
            Initial value for layer scaling. Used for backward compatibility.
        initializer_range (`float`, *optional*, defaults to 0.01):
            Standard deviation for weight initialization. Used for backward compatibility.
        num_filters (`int`, *optional*, defaults to 32):
            Number of filters in initial convolutional layer. Used for backward compatibility.
        downsampling_ratios (`List[int]`, *optional*, defaults to `[2, 2, 4, 5, 5, 8]`):
            Downsampling ratios for each encoder layer. Used for backward compatibility.
        depths (`List[int]`, *optional*, defaults to `[3, 3, 3, 3, 3, 3, 8]`):
            Number of ConvNeXt blocks at each encoder stage. Used for backward compatibility.
        hidden_act (`str`, *optional*, defaults to `"gelu"`):
            Activation function to use. Used for backward compatibility.
        ffn_expansion (`int`, *optional*, defaults to 4):
            Expansion factor for feed-forward networks. Used for backward compatibility.

    Example:

    ```python
    >>> from transformers import VibeVoiceAcousticTokenizerModel, VibeVoiceAcousticTokenizerConfig

    >>> # Initializing a VibeVoice Acoustic Tokenizer configuration
    >>> configuration = VibeVoiceAcousticTokenizerConfig()

    >>> # Initializing a model (with random weights)
    >>> model = VibeVoiceAcousticTokenizerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "vibevoice_acoustic_tokenizer"
    sub_configs = {
        "encoder_config": VibeVoiceAcousticTokenizerEncoderConfig,
        "decoder_config": VibeVoiceAcousticTokenizerDecoderConfig,
    }

    def __init__(
        self,
        encoder_config=None,
        decoder_config=None,
        vae_std=0.625,
        # BC for flat config parameters
        channels=1,
        hidden_size=64,
        kernel_size=7,
        rms_norm_eps=1e-5,
        layer_scale_init_value=1e-6,
        initializer_range=1e-2,
        num_filters=32,
        downsampling_ratios=[2, 2, 4, 5, 5, 8],
        depths=[3, 3, 3, 3, 3, 3, 8],
        hidden_act="gelu",
        ffn_expansion=4,
        **kwargs,
    ):
        if encoder_config is None:
            encoder_config = VibeVoiceAcousticTokenizerEncoderConfig(
                channels=channels,
                hidden_size=hidden_size,
                kernel_size=kernel_size,
                rms_norm_eps=rms_norm_eps,
                layer_scale_init_value=layer_scale_init_value,
                initializer_range=initializer_range,
                num_filters=num_filters,
                downsampling_ratios=downsampling_ratios,
                depths=depths,
                hidden_act=hidden_act,
                ffn_expansion=ffn_expansion,
            )
        elif isinstance(encoder_config, dict):
            encoder_config = VibeVoiceAcousticTokenizerEncoderConfig(**encoder_config)
        self.encoder_config = encoder_config

        if decoder_config is None:
            # NOTE (ebezzam/eustlb) add condition to deduce decoder config if encoder_config provided?
            decoder_config = VibeVoiceAcousticTokenizerDecoderConfig(
                channels=channels,
                hidden_size=hidden_size,
                kernel_size=kernel_size,
                rms_norm_eps=rms_norm_eps,
                layer_scale_init_value=layer_scale_init_value,
                initializer_range=initializer_range,
                num_filters=num_filters,
                upsampling_ratios=list(reversed(downsampling_ratios)),
                decoder_depths=list(reversed(depths)),
                hidden_act=hidden_act,
                ffn_expansion=ffn_expansion,
            )
        elif isinstance(decoder_config, dict):
            decoder_config = VibeVoiceAcousticTokenizerDecoderConfig(**decoder_config)
        self.decoder_config = decoder_config

        self.vae_std = vae_std
        super().__init__(**kwargs)

    @property
    def hop_length(self):
        return self.encoder_config.hop_length


__all__ = [
    "VibeVoiceAcousticTokenizerConfig",
    "VibeVoiceAcousticTokenizerEncoderConfig",
    "VibeVoiceAcousticTokenizerDecoderConfig",
]
