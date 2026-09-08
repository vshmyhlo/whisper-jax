# Whisper JAX

This repository contains optimised JAX code for OpenAI's [Whisper Model](https://arxiv.org/abs/2212.04356), largely built 
on the 🤗 Hugging Face Transformers Whisper implementation. Compared to OpenAI's PyTorch code, Whisper JAX runs over **70x** 
faster, making it the fastest Whisper implementation available.

The JAX code is compatible on CPU, GPU and TPU, and can be run standalone (see [Pipeline Usage](#pipeline-usage)) or 
as an inference endpoint (see [Creating an Endpoint](#creating-an-endpoint)).

For a quick-start guide to running Whisper JAX on a Cloud TPU, refer to the following Kaggle notebook, where we transcribe 30 mins of audio in approx 30 sec:

[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/sgandhi99/whisper-jax-tpu)

The Whisper JAX model is also running as a demo on the Hugging Face Hub:

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/sanchit-gandhi/whisper-jax)

## Installation

Whisper JAX requires Python 3.12 or newer and is tested with JAX 0.8.2. JAX is installed as a package dependency.
Accelerator users can install the appropriate JAX distribution first by following the official JAX installation guide:
https://docs.jax.dev/en/latest/installation.html

Once the appropriate version of JAX has been installed, Whisper JAX can be installed through pip:
```
pip install git+https://github.com/sanchit-gandhi/whisper-jax.git
```

To update the Whisper JAX package to the latest version, simply run:
```
pip install --upgrade --no-deps --force-reinstall git+https://github.com/sanchit-gandhi/whisper-jax.git
```

## Pipeline Usage

The recommended way of running Whisper JAX is through the [`FlaxWhisperPipline`](https://github.com/sanchit-gandhi/whisper-jax/blob/main/whisper_jax/pipeline.py#L57) abstraction class. This class handles all
the necessary pre- and post-processing, as well as wrapping the generate method for data parallelism across accelerator devices.

Whisper JAX makes use of JAX's [`pmap`](https://jax.readthedocs.io/en/latest/_autosummary/jax.pmap.html) function for data parallelism across GPU/TPU devices. This function is _Just In Time (JIT)_ 
compiled the first time it is called. Thereafter, the function will be _cached_, enabling it to be run in super-fast time:

```python
from whisper_jax import FlaxWhisperPipline

# instantiate pipeline
pipeline = FlaxWhisperPipline("openai/whisper-large-v2")

# JIT compile the forward call - slow, but we only do once
text = pipeline("audio.mp3")

# used cached function thereafter - super fast!!
text = pipeline("audio.mp3")
```

### Half-Precision

The model computation can be run in half-precision by passing the dtype argument when instantiating the pipeline. This will 
speed-up the computation quite considerably by storing intermediate tensors in half-precision. There is no change to the precision 
of the model weights.

For most GPUs, the dtype should be set to `jnp.float16`. For A100 GPUs or TPUs, the dtype should be set to `jnp.bfloat16`:
```python
from whisper_jax import FlaxWhisperPipline
import jax.numpy as jnp

# instantiate pipeline in bfloat16
pipeline = FlaxWhisperPipline("openai/whisper-large-v2", dtype=jnp.bfloat16)
```

On supported NVIDIA GPUs, JAX's cuDNN Flash Attention implementation can be enabled explicitly:

```python
pipeline = FlaxWhisperPipline(
    "openai/whisper-large-v2",
    dtype=jnp.bfloat16,
    attention_backend="cudnn",
)
```

The default backend uses native Flax attention with FP32 softmax and probabilities cast back to the computation dtype.
The model also falls back to it when attention weights are
requested or when attention dropout is active during training. The `"cudnn"` backend requires `jnp.float16` or
`jnp.bfloat16` computation and an NVIDIA CUDA-enabled JAX installation. Unsupported cuDNN shapes raise an error during
compilation.

FP16 and BF16 can choose different tokens. See the [measured generation parity](#generation-parity-and-precision)
below. `dtype` controls computation and KV-cache storage; parameters retain their loaded dtype. The separate
`shard_params()` API explicitly converts parameters to BF16.

### Batching
Whisper JAX also provides the option of _batching_ a single audio input across accelerator devices. The audio is first 
chunked into 30 second segments, and then chunks dispatched to the model to be transcribed in parallel. The resulting 
transcriptions are stitched back together at the boundaries to give a single, uniform transcription. In practice, batching 
provides a 10x speed-up compared to transcribing the audio samples sequentially, with a less than 1% penalty to the WER[^1], provided the batch size is selected large enough. 

To enable batching, pass the `batch_size` parameter when you instantiate the pipeline:

```python
from whisper_jax import FlaxWhisperPipline

# instantiate pipeline with batching
pipeline = FlaxWhisperPipline("openai/whisper-large-v2", batch_size=16)
```

### Task

By default, the pipeline transcribes the audio file in the language it was spoken in. For speech translation, set the 
`task` argument to `"translate"`:

```python
# translate
text = pipeline("audio.mp3", task="translate")
```

### Timestamps

The [`FlaxWhisperPipline`](https://github.com/sanchit-gandhi/whisper-jax/blob/main/whisper_jax/pipeline.py#L57) also supports timestamp prediction. Note that enabling timestamps will require a second JIT compilation of the 
forward call, this time including the timestamp outputs:

```python
# transcribe and return timestamps
outputs = pipeline("audio.mp3",  task="transcribe", return_timestamps=True)
text = outputs["text"]  # transcription
chunks = outputs["chunks"]  # transcription + timestamps
```

### Putting it all together
In the following code snippet, we instantiate the model in bfloat16 precision with batching enabled, and transcribe the audio file 
returning timestamps tokens: 

```python
from whisper_jax import FlaxWhisperPipline
import jax.numpy as jnp

# instantiate pipeline with bfloat16 and enable batching
pipeline = FlaxWhisperPipline("openai/whisper-large-v2", dtype=jnp.bfloat16, batch_size=16)

# transcribe and return timestamps
outputs = pipeline("audio.mp3",  task="transcribe", return_timestamps=True)
```

## Model Usage

The Whisper JAX model can use on a more granular level in much the same way as the original Hugging Face 
Transformers implementation. This requires the Whisper processor to be loaded separately to the model to handle the
pre- and post-processing, and the generate function to be wrapped using `pmap` by hand:

```python
import jax.numpy as jnp
from datasets import load_dataset
from flax.jax_utils import replicate
from flax.training.common_utils import shard
from jax import device_get, pmap
from transformers import WhisperProcessor

from whisper_jax import FlaxWhisperForConditionalGeneration

# load the processor and model
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
model, params = FlaxWhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-large-v2", dtype=jnp.bfloat16, _do_init=False,
)

def generate_fn(input_features):
    pred_ids = model.generate(
        input_features, task="transcribe", return_timestamps=False, max_length=model.config.max_length, params=params,
    )
    return pred_ids.sequences

# pmap the generate function for data parallelism
p_generate = pmap(generate_fn, "input_features")
# replicate the parameters across devices
params = replicate(params)

# load a dummy sample from the LibriSpeech dataset
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
sample = ds[0]["audio"]

# pre-process: convert the audio array to log-mel input features
input_features = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="np").input_features
# replicate the input features across devices for DP
input_features = shard(input_features)

# run the forward pass (JIT compiled the first time it is called)
pred_ids = p_generate(input_features)
output_ids = device_get(pred_ids.reshape(-1, model.config.max_length))

# post-process: convert tokens ids to text string
transcription = processor.batch_decode(pred_ids, skip_special_tokens=True)
```

## Available Models and Languages
All Whisper models on the Hugging Face Hub with Flax weights are compatible with Whisper JAX. This includes, but is not limited to,
the official OpenAI Whisper checkpoints:

| Size     | Parameters | English-only                                         | Multilingual                                        |
|----------|------------|------------------------------------------------------|-----------------------------------------------------|
| tiny     | 39 M       | [✓](https://huggingface.co/openai/whisper-tiny.en)   | [✓](https://huggingface.co/openai/whisper-tiny)     |
| base     | 74 M       | [✓](https://huggingface.co/openai/whisper-base.en)   | [✓](https://huggingface.co/openai/whisper-base)     |
| small    | 244 M      | [✓](https://huggingface.co/openai/whisper-small.en)  | [✓](https://huggingface.co/openai/whisper-small)    |
| medium   | 769 M      | [✓](https://huggingface.co/openai/whisper-medium.en) | [✓](https://huggingface.co/openai/whisper-medium)   |
| large    | 1550 M     | x                                                    | [✓](https://huggingface.co/openai/whisper-large)    |
| large-v2 | 1550 M     | x                                                    | [✓](https://huggingface.co/openai/whisper-large-v2) |

Should you wish to use a fine-tuned Whisper checkpoint in Whisper JAX, you should first convert the PyTorch weights to Flax.
This is straightforward through use of the `from_pt` argument, which will convert the PyTorch state dict to a frozen Flax 
parameter dictionary on the fly. You can then push the converted Flax weights to the Hub to be used directly in Flax 
the next time they are required. Note that converting weights from PyTorch to Flax requires both PyTorch and Flax to be installed.

For example, to convert the fine-tuned checkpoint [`sanchit-gandhi/whisper-small-hi`](https://huggingface.co/sanchit-gandhi/whisper-small-hi) from the blog post [Fine-Tuning Whisper](https://huggingface.co/blog/fine-tune-whisper):
```python
from whisper_jax import FlaxWhisperForConditionalGeneration, FlaxWhisperPipline
import jax.numpy as jnp

checkpoint_id = "sanchit-gandhi/whisper-small-hi"
# convert PyTorch weights to Flax
model = FlaxWhisperForConditionalGeneration.from_pretrained(checkpoint_id, from_pt=True)
# push converted weights to the Hub
model.push_to_hub(checkpoint_id)

# now we can load the Flax weights directly as required
pipeline = FlaxWhisperPipline(checkpoint_id, dtype=jnp.bfloat16, batch_size=16)
```

## Advanced Usage
More advanced users may wish to explore different parallelisation techniques. The Whisper JAX code is
built on-top of the [T5x codebase](https://github.com/google-research/t5x), meaning it can be run using model, activation, and data parallelism using the T5x 
partitioning convention. To use T5x partitioning, the logical axis rules and number of model partitions must be defined.
For more details, the user is referred to the official T5x partitioning guide: https://github.com/google-research/t5x/blob/main/docs/usage/partitioning.md

### Pipeline
The following code snippet demonstrates how data parallelism can be achieved using the pipeline `shard_params` method in 
an entirely equivalent way to `pmap`:

```python
from whisper_jax import FlaxWhisperPipline
import jax.numpy as jnp

# 2D parameter and activation partitioning for DP
logical_axis_rules_dp = (
    ("batch", "data"),
    ("mlp", None),
    ("heads", None),
    ("vocab", None),
    ("embed", None),
    ("embed", None),
    ("joined_kv", None),
    ("kv", None),
    ("length", None),
    ("num_mel", None),
    ("channels", None),
)

pipeline = FlaxWhisperPipline("openai/whisper-large-v2", dtype=jnp.bfloat16, batch_size=16)
pipeline.shard_params(num_mp_partitions=1, logical_axis_rules=logical_axis_rules_dp)
```

### Model
It is also possible to use the Whisper JAX model with T5x partitioning by defining a T5x inference state and T5x partitioner:

```python
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import freeze
from jax.sharding import PartitionSpec as P

from whisper_jax import FlaxWhisperForConditionalGeneration, InferenceState, PjitPartitioner


# 2D parameter and activation partitioning for DP
logical_axis_rules_dp = [
    ("batch", "data"),
    ("mlp", None),
    ("heads", None),
    ("vocab", None),
    ("embed", None),
    ("embed", None),
    ("joined_kv", None),
    ("kv", None),
    ("length", None),
    ("num_mel", None),
    ("channels", None),
]

model, params = FlaxWhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-large-v2",
    _do_init=False,
    dtype=jnp.bfloat16,
)


def init_fn():
    input_shape = (1, 80, 3000)

    input_features = jnp.zeros(input_shape, dtype="f4")
    input_features = input_features.at[(..., -1)].set(model.config.eos_token_id)

    decoder_input_ids = jnp.zeros((input_shape[0], 1), dtype="i4")
    decoder_attention_mask = jnp.ones_like(decoder_input_ids)

    batch_size, sequence_length = decoder_input_ids.shape
    decoder_position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

    rng = jax.random.PRNGKey(0)
    init_params = model.module.init(
        rng,
        input_features=input_features,
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=decoder_attention_mask,
        decoder_position_ids=decoder_position_ids,
        return_dict=False,
    )
    return init_params


# Axis names metadata
param_axes = jax.eval_shape(init_fn)["params_axes"]

# Create InferenceState, since the partitioner expects it
state = InferenceState(
    step=jnp.array(0),
    params=freeze(model.params_shape_tree),
    params_axes=freeze(param_axes),
    flax_mutables=None,
    flax_mutables_axes=param_axes,
)

# Define the pjit partitioner with 1 model partition
partitioner = PjitPartitioner(
    num_partitions=1,
    logical_axis_rules=logical_axis_rules_dp,
)

mesh_axes = partitioner.get_mesh_axes(state)
params_spec = mesh_axes.params

p_shard_params = partitioner.partition(model.to_bf16, (params_spec,), params_spec)


def generate(params, input_features):
    output_ids = model.generate(input_features, params=params, max_length=model.config.max_length).sequences
    return output_ids


p_generate = partitioner.partition(
    generate,
    in_axis_resources=(params_spec, P("data")),
    out_axis_resources=P("data"),
)

# This will auto-magically run in mesh context
params = p_shard_params(freeze(params))

# you can now run the forward pass with: 
# pred_ids = p_generate(input_features)
```

## Generation parity and precision

The [recorded real-audio run](benchmarks/reports/real_audio_parity.json) compares JAX against unmodified
[OpenAI Whisper v20250625](https://github.com/openai/whisper/tree/v20250625), using its original `tiny.en` and `tiny`
checkpoints, JFK audio, and two LibriSpeech clips. Each clip is decoded with and without timestamps: **12 cases per dtype**.
Checkpoint/audio hashes, source URLs, complete token sequences, transcripts, and per-case errors are recorded.

The Torch reference keeps **FP32 stored parameters and the original `DecodingOptions.fp16=True` default** throughout.
It is never converted to BF16 or called with `model.half()`. This tests native single-window greedy `decode()`, including
suppression rules, timestamps and EOS; it does not use `transcribe()`'s CPU FP32 fallback. JAX also keeps FP32 parameters.
Normalization/softmax use FP32 internally, while projections, activations and self-attention KV caches use the requested
computation dtype. Expected outputs are generated by Torch in a separate process. The JAX process imports no Torch,
processes the real waveforms using NumPy audio features, and explicitly calls `jax.jit(...).lower(...).compile()` for
encoding, cached decoding, and complete generation. Disabling JIT makes validation fail.

Measured on ARM CPU, JAX 0.8.2 / PyTorch 2.14.0 (2026-09-08):

| JAX computation / parameters | Exact token sequences | Exact transcripts | Worst logit MAE | Maximum absolute logit error | Worst relative L2 logit error |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP32 / FP32 | 12/12 | 12/12 | 0.018817 | 0.090569 | 0.001224 |
| FP16 / FP32 | 12/12 | 12/12 | 0.020977 | 0.093750 | 0.002057 |
| BF16 / FP32 | **9/12** | **9/12** | 0.141118 | 0.667969 | 0.015449 |

| JAX computation | Worst encoder MAE | Maximum absolute encoder error | Worst relative L2 encoder error |
| --- | ---: | ---: | ---: |
| FP32 | 0.001742 | 0.902971 | 0.004063 |
| FP16 | 0.002822 | 1.123047 | 0.004835 |
| BF16 | 0.017435 | 5.066406 | 0.029955 |

“Worst” means the largest per-case metric, not an average across clips. Raw logits are compared over the full vocabulary
at every native reference-prefix decision, including EOS, through JAX's KV cache. Using the same reference prefix isolates
numerical error from divergent generated text. Generation itself runs freely and is compared separately. These errors
measure agreement with Torch, not accuracy against human transcripts.

**BF16 does not provide exact Torch parity.** In this run, `tiny.en` inserts a comma after “belly” on LibriSpeech clip 2
with timestamps; `tiny` inserts a comma after “you” on JFK without timestamps; and `tiny` changes “flower-fat and sauce”
to “flower-fatten sauce” on LibriSpeech clip 1 with timestamps, also moving its closing timestamp from 10.16 to 10.12 s.
FP16 matches all tested outputs; this is evidence for these fixtures, not a guarantee for every recording or accelerator.

### Reproducing the tests

Prepare the reference once in a separate development environment containing this repository, `openai-whisper==20250625`,
PyTorch and system `ffmpeg`. Downloads and large artifacts stay outside the repository:

```sh
PYTHONPATH=. python tests/fixtures/generate_real_whisper_reference.py --output /tmp/whisper-real-parity
```

In a JAX environment without PyTorch installed, run the complete suite, including the real checkpoint tests and
two-replica CPU HLO audit:

```sh
XLA_FLAGS=--xla_force_host_platform_device_count=2 \
  WHISPER_REAL_PARITY_DIR=/tmp/whisper-real-parity PYTHONPATH=. \
  python -m unittest discover -s tests -v
```

Real-data tests skip explicitly when `WHISPER_REAL_PARITY_DIR` is absent. They require exact FP32/FP16 token parity and
bound BF16 numerical error and each case's known token drift. New BF16 token differences fail for previously exact cases;
improvements are allowed. Offline tests additionally compare original whisper-jax and Torch fixtures and exercise cache
prefill/chunks, padding, beam-cache reordering, forced prompts, timestamps, and cuDNN dispatch. See
[fixture provenance](tests/fixtures/README.md).

To write a fresh numerical report and optimized HLO:

```sh
XLA_FLAGS=--xla_force_host_platform_device_count=2 PYTHONPATH=. \
  python benchmarks/check_real_audio_parity.py --reference-dir /tmp/whisper-real-parity \
  --report /tmp/whisper-parity-report.json --hlo-dir /tmp/whisper-hlo
```

Add `--dtypes float32 float16 --strict-tokens` to make any token mismatch an error. Including BF16 with `--strict-tokens`
currently exits nonzero. On accelerators use actual devices instead of the CPU `XLA_FLAGS` setting.

### Compiled execution and memory review

The report includes a complete encoder/decoder forward pass compiled with `pmap`, replicated FP32 weights and BF16
computation on two logical CPU devices. Different inputs on each replica match separately compiled single-device results
exactly. HLO contains **zero all-reduce, all-gather, reduce-scatter, all-to-all or collective-permute operations**, and no
host callbacks. Runtime model code uses native JAX/Flax; Torch is confined to reference-generation tooling.

Complete generation has one XLA token loop. Its eight self-attention key/value buffers use BF16 and total 576 KiB per
example at `max_length=96`. XLA hoists all eight cross-attention K/V projections outside that loop: they are computed once
and reused, although the Python decoder expresses the projections on each call. A separately compiled standalone
`decode()` call does not benefit from that cross-step hoisting.

The review replaced one-hot embedding matrix products with gathers, preserving padding and sharded-vocabulary semantics.
This removes three embedding matrix products from the generation HLO, including a full-vocabulary operation per generated
token. For `tiny.en`, compiled generation temporary allocation falls from **323.73 MiB to 247.10 MiB (23.7%)** with unchanged
outputs from the embedding replacement ([isolated comparison](benchmarks/reports/embedding_hlo_comparison.json)).
Replicated FP32 weights/input arguments occupy approximately 145 MiB per device.
Forward temporary allocation is 138.82 MiB for a one-token `tiny.en` prompt and 244.61 MiB for a three-token `tiny` prompt.
These are XLA's executable memory estimates, not measured total process/device peak memory.

CPU HLO expands BF16 arithmetic to FP32 and has many conversion operations; this cannot establish GPU kernel performance.
The default attention path also materializes attention matrices. GPU cuDNN execution, accelerator memory/timing, and larger
checkpoints remain unverified here. Dispatch/cache tests use an XLA stand-in for cuDNN. Further accelerator work should
measure cuDNN mask/layout costs and compare the existing one-hot cache writes against dynamic slice updates. Real-audio
coverage here is greedy English single-window decoding; long-form stitching, language detection, translation, sampling and
beam-search parity against native Torch are outside this measured matrix.

## Benchmarks

We compare Whisper JAX to the official [OpenAI implementation](https://github.com/openai/whisper) and the [🤗 Transformers 
implementation](https://huggingface.co/docs/transformers/model_doc/whisper). We benchmark the models on audio samples of 
increasing length and report the average inference time in seconds over 10 repeat runs. For all three systems, we pass a 
pre-loaded audio file to the model and measure the time for the forward pass. Leaving the task of loading the audio file 
to the systems adds an equal offset to all the benchmark times, so the actual time for loading **and** transcribing an 
audio file will be higher than the reported numbers.

OpenAI and Transformers both run in PyTorch on GPU. Whisper JAX runs in JAX on GPU and TPU. OpenAI transcribes the audio 
sequentially in the order it is spoken. Both Transformers and Whisper JAX use a batching algorithm, where chunks of audio 
are batched together and transcribed in parallel (see section [Batching](#batching)).

**Table 1:** Average inference time in seconds for audio files of increasing length. GPU device is a single A100 40GB GPU. 
TPU device is a single TPU v4-8.

<div align="center">

|           | OpenAI  | Transformers | Whisper JAX | Whisper JAX |
|-----------|---------|--------------|-------------|-------------|
|           |         |              |             |             |
| Framework | PyTorch | PyTorch      | JAX         | JAX         |
| Backend   | GPU     | GPU          | GPU         | TPU         |
|           |         |              |             |             |
| 1 min     | 13.8    | 4.54         | 1.72        | 0.45        |
| 10 min    | 108.3   | 20.2         | 9.38        | 2.01        |
| 1 hour    | 1001.0  | 126.1        | 75.3        | 13.8        |
|           |         |              |             |             |

</div>

## Creating an Endpoint

The Whisper JAX model is running as a demo on the Hugging Face Hub:

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/sanchit-gandhi/whisper-jax)

However, at peak times there may be a queue of users that limit how quickly your audio input is transcribed. In this case,
you may benefit from running the model yourself, such that you have unrestricted access to the Whisper JAX model.

If you are just interested in running the model in a standalone Python script, refer to the Kaggle notebook Whisper JAX TPU:

[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/sgandhi99/whisper-jax-tpu)

Otherwise, we provide all the necessary code for creating an inference endpoint. To obtain this code, first clone the 
repository on the GPU/TPU on which you want to host the endpoint:
```
git clone https://github.com/sanchit-gandhi/whisper-jax
```

And then install Whisper JAX from source, with the required additional endpoint dependencies:
```
cd whisper-jax
pip install -e .["endpoint"]
```

We recommend that you set-up an endpoint in the same zone/region as the one you are based in. This reduces the communication 
time between your local machine and the remote one, which can significantly reduce the overall request time.

### Gradio App

The Python script [`app.py`](app/app.py) contains the code to launch a Gradio app with the Whisper large-v2 model.
By default, it uses a batch size of 16 and bfloat16 half-precision. You should update these parameters depending on your 
GPU/TPU device (as explained in the sections on [Half-precision](#half-precision) and [Batching](#batching)).

We can launch the Gradio app on port 7860 (default) on our GPU/TPU device through the following command:
```
python app/app.py
```

This will launch a Gradio demo with the same interface as the official Whisper JAX demo. To view the Gradio app remotely, 
we have two options:

1. Open the port 7860 on the GPU/TPU device to listen to all requests
2. Start an ngrok server on the GPU/TPU that redirects requests to port 7860

To open the port 7860 on your GPU/TPU, refer to your hardware provider's firewall instructions (for GCP, these can be 
found [here](https://cloud.google.com/firewall/docs/using-firewalls)). Once you have opened port 7860, you should be able 
to access the gradio demo through the http address:
```
http://DEVICE-IP:7860
```
where `DEVICE-IP` is the public IP address of your GPU/TPU. We can verify this address is accessible by opening this 
http address in a browser window on our local machine.

Alternatively, we can direct network requests to the Gradio app using ngrok. By using ngrok, we don't need to open the 
port 7860 on our GPU/TPU - ngrok will provide us with a public http address that will automatically redirect requests to 
port 7860 on our accelerator. However, in our experience, using ngrok was less reliable than a direct tunnel to port 7860, 
thus we recommend option 1 here where possible.

To set-up ngrok on your GPU/TPU, first install ngrok according to the official [installation guide](https://ngrok.com/download).
You should authenticate your ngrok account if you have one, otherwise your ngrok server will be time-limited to 2 hours.
Once installed and authenticated, you can launch an ngrok server on port 7860:
```
ngrok http 7860
```
The ngrok http address will be of the form:
```
https://NGROK-ADDRESS.ngrok.io
```
which can be used to access the Gradio demo through a web browser.

### Sending Requests

Independent of whether you've chosen to open the port 7860 or use ngrok, we're now ready to send audio file requests to our
endpoint. To do this, we'll make use of the `gradio_client` library. If you already have a recent version of Gradio, 
then the `gradio_client` library is included as a dependency.

Otherwise, the lightweight `gradio_client` package can be installed from pip and is tested to work with Python versions 
3.9 or higher:
```
pip install --upgrade gradio_client
```

We can now send json requests to our endpoint using ngrok. The function `transcribe_audio` sends an audio file to our endpoint 
and returns the transcription:

```python
from gradio_client import Client

# make sure this URL matches your http web address
API_URL = "http://DEVICE-IP:7860/" # if using port 7860
API_URL = "https://NGROK-ADDRESS.ngrok.io/" # if using ngrok

# set up the Gradio client
client = Client(API_URL)

def transcribe_audio(audio_path, task="transcribe", return_timestamps=False):
    """Function to transcribe an audio file using our endpoint"""
    text, runtime = client.predict(
        audio_path,
        task,
        return_timestamps,
        api_name="/predict_1",
    )
    return text

# transcribe an audio file using our endpoint
output = transcribe_audio("audio.mp3")

# transcribe with timestamps
output_with_timestamps = transcribe_audio("audio.mp3", return_timestamps=True)
```

## Acknowledgements

* 🤗 Hugging Face Transformers for the base Whisper implementation, particularly to [andyehrenberg](https://github.com/andyehrenberg) for the [Flax Whisper PR](https://github.com/huggingface/transformers/pull/20479) and [ArthurZucker](https://github.com/ArthurZucker) for the batching algorithm 
* Gradio for their easy-to-use package for building ML demos, and [pcuenca](https://github.com/pcuenca) for the help in hooking the demo up to the TPU 
* Google's [TPU Research Cloud (TRC)](https://sites.research.google/trc/about/) programme for Cloud TPUs
* Google's [t5x Repository](https://github.com/google-research/t5x) for the model partitioning framework

[^1]: See WER results from Colab: https://colab.research.google.com/drive/1rS1L4YSJqKUH_3YxIQHBI982zso23wor?usp=sharing
