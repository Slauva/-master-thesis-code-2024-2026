# ARL EEGModels Attribution

The PyTorch modules in `models.py` are spectral-input adaptations of architectures from:

- Vernon J. Lawhern et al., "EEGNet: a compact convolutional neural network for EEG-based
  brain-computer interfaces", Journal of Neural Engineering, 2018.
- Robin Tibor Schirrmeister et al., "Deep learning with convolutional neural networks for EEG
  decoding and visualization", Human Brain Mapping, 2017.
- Nicholas Waytowich et al., "Compact convolutional neural networks for classification of
  asynchronous steady-state visual evoked potentials", Journal of Neural Engineering, 2018.
- Upstream implementation: <https://github.com/vlawhern/arl-eegmodels>

The original implementation targets raw EEG tensors in Keras/TensorFlow. This project changed the
input convention to `(batch, spectral_planes, electrodes, spectral_time)`, emits 36 independent
logits for multi-label reconstruction, caps kernels at the available spectral width, and uses
adaptive global pooling where the original flatten geometry is not valid for short spectral axes.
These modules are therefore adaptations, not numerically identical translations.

EEGNet-v1's original Keras layer-local L1/L2 regularizers are not embedded in the PyTorch module.
The shared training workflow applies its explicitly configured optimizer weight decay instead.

The complete upstream CC0 1.0 and Apache License 2.0 text is retained in
`ARL_EEGMODELS_LICENSE.txt`.
