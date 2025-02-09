import os
import torch
import argparse
from zact import ZACT
from tqdm import tqdm
import soundfile as sf
from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download
from zact.models.facodec import FACodecEncoder, FACodecDecoder
from torch.utils.data import Dataset, DataLoader
import numpy as np


SR = 16000

class TestDataset(Dataset):
    def __init__(self, input_dir, text_file):
        super().__init__()
        self.samples = []
        total_dur = 0
        with open(text_file, 'r') as fin:
            for line in tqdm(list(fin)):
                target_name, prompt_name, target_transcript, _, _, dur  = line.rstrip().split('|')
                prompt_filepath = os.path.join(input_dir, prompt_name)
                self.samples.append({
                    "target_name": target_name.split(".")[0],
                    "target_transcript": target_transcript,
                    "prompt_filepath": prompt_filepath
                })
                total_dur += float(dur)
            print("Total duration: {:.2f}".format(total_dur/3600))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, index):
        sample = self.samples[index]
        return sample["target_name"], sample["target_transcript"], sample["prompt_filepath"]

def get_codec():
    fa_encoder = FACodecEncoder(
        ngf=32,
        up_ratios=[2, 4, 5, 5],
        out_channels=256,
    )

    fa_decoder = FACodecDecoder(
        in_channels=256,
        upsample_initial_channel=1024,
        ngf=32,
        up_ratios=[5, 5, 4, 2],
        vq_num_q_c=2,
        vq_num_q_p=1,
        vq_num_q_r=3,
        vq_dim=256,
        codebook_dim=8,
        codebook_size_prosody=10,
        codebook_size_content=10,
        codebook_size_residual=10,
        use_gr_x_timbre=True,
        use_gr_residual_f0=True,
        use_gr_residual_phone=True,
    )

    encoder_ckpt = hf_hub_download(
        repo_id="amphion/naturalspeech3_facodec", 
        filename="ns3_facodec_encoder.bin"
    )
    decoder_ckpt = hf_hub_download(
        repo_id="amphion/naturalspeech3_facodec", 
        filename="ns3_facodec_decoder.bin"
    )
    fa_encoder.load_state_dict(torch.load(encoder_ckpt))
    fa_decoder.load_state_dict(torch.load(decoder_ckpt))
    fa_encoder.eval()
    fa_decoder.eval()
    
    return fa_encoder, fa_decoder


def synthesize(args):
    text_file = args.text_file
    input_dir = args.input_dir
    output_dir = args.output_dir
    ckpt_path = args.ckpt_path
    cfg_path = args.cfg_path
    device = args.device
    temperature = float(args.temp)
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if not os.path.exists(os.path.join(output_dir, 'synth')):
        os.mkdir(os.path.join(output_dir, 'synth'))

    if not os.path.exists(os.path.join(output_dir, 'prior')):
        os.mkdir(os.path.join(output_dir, 'prior'))
    
    cfg = OmegaConf.load(cfg_path)
    codec_encoder, codec_decoder = get_codec()
    model = ZACT.from_pretrained(
        cfg=cfg,
        ckpt_path=ckpt_path,
        device=device,
        training_mode=False
    )

    dataset = TestDataset(input_dir=input_dir, text_file=text_file)
    loader = DataLoader(dataset, batch_size=1)
    
    infer_times = []
    total_samples = []
    for batch in tqdm(loader):
        filename, transcript, prompt_filepath = batch
        filename, transcript, prompt_filepath = filename[0], transcript[0], prompt_filepath[0]
        output = model.synthesize(
            text=transcript,
            acoustic_prompt=prompt_filepath,
            codec_encoder=codec_encoder,
            codec_decoder=codec_decoder,
            temperature=temperature,
        )
 
        synth_wav = output['synth_wav']
        prior_wav = output['prior_wav']
        infer_times.append(output['time'])

        sf.write(
            file=os.path.join(output_dir, 'synth', f'{filename}.wav'), 
            data=synth_wav,
            samplerate=SR
        )
    
    total_infer_time = sum(infer_times)
    num_samples = len(infer_times)
    avg_infer_time = total_infer_time / len(infer_times)
    return total_infer_time, num_samples, avg_infer_time


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_file', required=True)
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--ckpt_path', required=True)
    parser.add_argument('--cfg_path', required=True)
    parser.add_argument('--temp', default='0.01')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    total_infer_time, num_samples, avg_infer_time = synthesize(args)
    
    print('='*100)
    print('Number of samples: ', num_samples)
    print('Total inference time: ', total_infer_time)
    print('Average inference time: ', total_infer_time)
    print('='*100)