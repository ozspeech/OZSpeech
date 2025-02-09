import os
import re
import time
import torch
import librosa
import numpy as np
from g2p_en import G2p
from string import punctuation
from omegaconf import DictConfig
from zact.text import text_to_sequence
from transformers import BertTokenizer
from zact.models.zact_lightning import ZACTLightning
from zact.models.synthesizer import (
    CodesGenerator,
    FlowMatching,
)
from zact.models.facodec import (
    FACodecEncoder,
    FACodecDecoder,
)


class ZACT(ZACTLightning):
    
    @classmethod
    def from_pretrained(cls, cfg, ckpt_path, device, training_mode=False):
        cfg['flow_matching']['device'] = device
        cfg['codes_generator']['device'] = device
        model = ZACT(cfg)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['state_dict'])
        del ckpt
        if not training_mode:
            model.eval()
        return model
    
    def __init__(self, cfg):
        super(ZACT, self).__init__()

        self.codes_generator = CodesGenerator(cfg['codes_generator'])
        self.flow_matching = FlowMatching(cfg['flow_matching'])
        
    def forward(
        self,
        x, 
        x_len, 
        y, 
        y_len, 
        durs,
        prompts,
        ):
        
        # Forward & compute losses of codes generator
        (
            prior, # (b, n_quantizes, length, hidden)
            logits, # (b, vocab_size + 1, n_quantizes, length)
            log_duration_predictions,
            duration_rounded,
            src_masks,
            tgt_masks,
            src_lens,
            tgt_lens 
        ) = self.codes_generator(
            texts=x,
            src_lens=x_len,
            max_src_len=x.shape[-1],
            tgt_lens=y_len,
            max_tgt_len=y.shape[-1],
            d_targets=durs,
        )
        codes_gen_losses = self.codes_generator.compute_loss(
            codes_pred=logits,
            codes=y,
            log_durations_pred=log_duration_predictions,
            durations=durs,
            src_masks=src_masks,
        )
        
        # Forward & compute losses of flow matching
        flow_losses = self.flow_matching.compute_loss(
            prior=prior,
            x1_tgt=y,
            x_len=y_len,
            x_max_len=y.shape[-1],
            prompts=prompts,
        )
        
        return codes_gen_losses | flow_losses
    
    @torch.inference_mode()
    def synthesize(
        self, 
        text: str,
        acoustic_prompt: str | np.ndarray | torch.Tensor,
        sr: int = 16000,
        codec_cfg: DictConfig = None,
        codec_encoder: torch.nn.Module = None,
        codec_decoder: torch.nn.Module = None,
        temperature: float = 0.02,
        lexicon_path: str = None,
        cleaners: str = ['english_cleaners'],
        ):
        
        if codec_encoder is None or codec_decoder is None:
            if codec_cfg is None:
                raise ValueError('The codec_encoder or codec_decoder is set to None. To initialize the codec encoder or decoder, you need to provide a codec_cfg of type omegaconf.DictConfig.')
            codec_cfg['encoder'] = self.device
            codec_cfg['decoder'] = self.device
            codec_encoder, codec_decoder = self._get_codec_models(codec_cfg)
            
        # get starting timestamp of the progress
        start_time = time.time()
        
        # process acoustic prompt
        acoustic_prompt = self._preprocess_acoustic_prompt(acoustic_prompt, sr)
        enc_out = codec_encoder(acoustic_prompt)
        _, prompt, _, _, timbre = codec_decoder(enc_out, eval_vq=False, vq=True)
        prompt = prompt.permute(1, 0, 2)
        
        # process phoneme
        phonemes, _, _ = self._preprocess_english(text, lexicon_path, cleaners)
        phonemes = phonemes.to(self.device)
        codes_generator_outputs = self.codes_generator(
            texts=phonemes,
            src_lens=torch.zeros(phonemes.size(0), device=self.device) + phonemes.size(1),
            max_src_len=phonemes.shape[-1],
        )
        prior, prior_logits = codes_generator_outputs[0], codes_generator_outputs[1]
        
        # flow matching euler solving
        logits = self.flow_matching.sampling(
            prior=prior,
            x_len=torch.zeros(prior.size(0), device=self.device) + prior.size(2),
            x_max_len=prior.size(2),
            prompts=prompt,
            temperature=temperature,
        )['logits']
        
        # revert codes into waveform
        prior_codes = prior_logits.softmax(1).argmax(1)
        prior_codes = prior_codes.permute(1, 0, 2)
        prior_embs = codec_decoder.vq2emb(prior_codes)
        prior_wav = codec_decoder.inference(prior_embs, timbre)
        prior_wav = prior_wav[0][0].detach().cpu().numpy()
        
        codes = logits.softmax(1).argmax(1)
        codes = codes.permute(1, 0, 2)
        embs = codec_decoder.vq2emb(codes)
        wav = codec_decoder.inference(embs, timbre)
        wav = wav[0][0].detach().cpu().numpy()
        
        # get ending time of the progress
        end_time = time.time()
                
        return {
            'synth_wav': wav,
            'prior_wav': prior_wav,
            'time': round(end_time - start_time, 3),
        }
    
    def _preprocess_acoustic_prompt(self, acoustic_prompt, sr=16000):
        if isinstance(acoustic_prompt, str):
            acoustic_prompt = librosa.load(acoustic_prompt, sr=sr)[0]
            acoustic_prompt = torch.from_numpy(acoustic_prompt).float()
            acoustic_prompt = acoustic_prompt.unsqueeze(0).unsqueeze(0).to(self.device)
        elif isinstance(acoustic_prompt, np.ndarray):
            acoustic_prompt = torch.from_numpy(acoustic_prompt).float()
            acoustic_prompt = acoustic_prompt.unsqueeze(0).unsqueeze(0).to(self.device)
        elif isinstance(acoustic_prompt, torch.Tensor):
            acoustic_prompt = acoustic_prompt.to(self.device)
        else:
            raise ValueError('Acoustic prompt must be one of [str, np.ndarray, torch.tensor]!')
        return acoustic_prompt
    
    def _get_text_tokenizer(self, text_tokenier=None):
        if text_tokenier:
            return BertTokenizer.from_pretrained(text_tokenier)
        else:
            return BertTokenizer.from_pretrained("google/bert_uncased_L-4_H-512_A-8")
    
    def _get_codec_models(self, codec_cfg):
        codec_encoder = FACodecEncoder.from_pretrained(codec_cfg['encoder']).eval()
        codec_decoder = FACodecDecoder.from_pretrained(codec_cfg['decoder']).eval()
        return codec_encoder, codec_decoder
    
    def _read_lexicon(self, lexicon_path=None):
        if not lexicon_path:
            lexicon_path = os.path.join(os.path.dirname(__file__), '..', 'lexicon', 'librispeech-lexicon.txt')
        lexicon = {}
        with open(lexicon_path) as f:
            for line in f:
                temp = re.split(r"\s+", line.strip("\n"))
                word = temp[0]
                phones = temp[1:]
                if word.lower() not in lexicon:
                    lexicon[word.lower()] = phones
        return lexicon
    
    def _preprocess_english(
        self, 
        text, 
        lexicon_path=None, 
        cleaners='english_cleaners'
        ):  
        text = text.rstrip(punctuation)
        lexicon = self._read_lexicon(lexicon_path)
        g2p = G2p()
        phones = []
        words = re.split(r"([,;.\-\?\!\s+])", text)
        for w in words:
            if w.lower() in lexicon:
                phones += lexicon[w.lower()]
            else:
                phones += list(filter(lambda p: p != " ", g2p(w)))
        phones = "{" + "}{".join(phones) + "}"
        phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
        phones = phones.replace("}{", " ")
        sequence = np.array(text_to_sequence(phones, cleaners))
        sequence = torch.from_numpy(sequence).unsqueeze(0).to(self.device)
        return sequence, text, phones