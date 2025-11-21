# mlx_lm.convert

> 2025-11-20

MLX natively supports quantization, a compression approach which reduces the
memory footprint of a language model by using a lower precision for storing the
parameters of the model. Using mlx_lm.convert, a model downloaded from Hugging
Face can be quantized in a few seconds. For example quantizing a 7B Mistral
model to a 4-bit takes only few seconds by running a simple command.

```bash
mlx_lm.convert \
  --hf-path mistralai/Mistral-7B-Instruct-v0.3 \
  -q \
  --upload-repo mlx-community/Mistral-7B-Instruct-v0.3-4bit
```

See https://machinelearning.apple.com/research/exploring-llms-mlx-m5