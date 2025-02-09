# OZSpeech: One-step Zero-shot Speech Synthesis with Learned-Prior-Conditioned Flow Matching

![Overall Architecture](https://github.com/ozspeech/OZSpeech/blob/main/figs/overall.png)

### TL;DR

OZSpeech is a cutting-edge Zero-Shot TTS model that utilizes Optimal Transport Flow Matching for one-step sampling, significantly reducing inference time while delivering high-quality synthesized speech.

# Installation requirement

Prepare your environment by creating a conda setup, preferably on Linux. Then, install the necessary requirements using pip:

```
pip install -r requirements.txt
```

If you plan to train the model yourself, a GPU is advised. However, you can still generate samples using our pretrained models without a GPU.

# Inference

### Download pretrained weights

To perform inference with pretrained weights, you must download the pretrained weights for both FaCodec and OZSpeech.

* With FaCodec, you can download the FaCodec Encoder and FaCodec Decoder directly from Hugging Face: [FaCodec Encoder](https://huggingface.co/amphion/naturalspeech3_facodec/blob/main/ns3_facodec_encoder.bin), [FaCodec Decoder](https://huggingface.co/amphion/naturalspeech3_facodec/blob/main/ns3_facodec_decoder.bin). Alternatively, you can access them via Google Drive using [this link](https://drive.google.com/drive/folders/1kk_TwwuzW8fViW6UDHdZW-JcsxP0mVOS?usp=drive_link).
* With OZSpeech, please refer [this link](https://drive.google.com/drive/u/0/folders/1XVjNHNvPQ6KF87i0mG2GDm4MTREXpp-o). You need to download both pretrained weights and config file for initializing model.

### Inference using python script

Script `synthesize.py` provides end-to-end pipeline for inference. Please follow the instructions:

```
python synthesize.py \
	--text_file path/to/manifest.txt \
	--input_dir path/to/dir/of/prompt/audio/files \
	--output_dir path/to/dir/for/output/audio/files \
	--ckpt_path path/to/ckpt.pt \
	--cfg_path path/to/config.yaml \
	--device cuda:0 # cpu as default
```

The format of manifest.txt file is as follow:

```
<groundtruth_filename>|<prompt_filename>|<groundtruth_transcription>|<prompt_transcription>|<prompt_transcription_clipped>|<groundtruth_duration>
```

The `LibriSpeech-test-clean` dataset was utilized to synthesize and evaluate our model. For simplicity, `<prompt_transcription>`, `<prompt_transcription_clipped>`, and `<groundtruth_duration>` can be disregarded. In this context, `<groundtruth_transcription>` represents the target content of the synthesized speech, while the audio file identified by `<prompt_filename>` serves as the input prompt. Additionally, we provide manifest files and prompt samples, which can be accessed via [this link](https://drive.google.com/drive/folders/1VqXmkPV73PqBxfe211iB5nItVNoJPv26?usp=drive_link).

### Inference in IPython notebook

The `synthesize.ipynb` notebook offers a user-friendly interface for inference. You can directly provide a single pair consisting of the prompt audio file path and the target text for synthesis. Give it a try!

# Training OZSpeech from scratch

TBD.

# Disclaimer

Any organization or individual is prohibited from using any technology mentioned in this paper to generate or edit someone's speech without his/her consent, including but not limited to government leaders, political figures, and celebrities. If you do not comply with this item, you could be in violation of copyright laws.
