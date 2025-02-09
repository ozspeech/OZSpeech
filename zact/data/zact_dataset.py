import os
import tgt
import json
import torch
import random
import numpy as np
from typing import Any, Dict, Optional
from lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader
from zact.text import text_to_sequence


class ZACTDataset(LightningDataModule):
    def __init__(self, config):
        super().__init__()

        # this line allows to access init params with 'self' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        
        self.name = config['name']
        self.data_root = config['data_root']
        self.train_manifest = config['train_manifest']
        self.valid_manifest = config['valid_manifest']
        self.sampling_rate = config['sampling_rate']
        self.dur_min = config['dur_min']
        self.dur_max = config['dur_max']
        self.n_words_min = config['n_words_min']
        self.prompt_dur_max = config['prompt_dur_max']
        self.prompt_reduced_factor = config['prompt_reduced_factor']
        self.down_factors = config['down_factors']
        self.vocab_size = config['vocab_size']
        self.batch_size = config['batch_size']
        self.num_workers = config['num_workers']
        self.pin_memory = config['pin_memory']
        self.cleaners = config['cleaners']
        self.add_blank = config['add_blank']
        self.seed = config['seed']
        self.device = config['device']
        self.sil_phones = config['sil_phones']

    def setup(self, stage: Optional[str] = None):  # pylint: disable=unused-argument
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        
        self.trainset = TextCodesDataset(  # pylint: disable=attribute-defined-outside-init
            self.data_root,
            self.train_manifest,
            self.cleaners,
            self.dur_min,
            self.dur_max,
            self.n_words_min,
            self.prompt_dur_max,
            self.sampling_rate,
            self.down_factors,
            self.sil_phones,
            self.add_blank,
            self.seed,
        )
        self.validset = TextCodesDataset(  # pylint: disable=attribute-defined-outside-init
            self.data_root,
            self.valid_manifest,
            self.cleaners,
            self.dur_min,
            self.dur_max,
            self.n_words_min,
            self.prompt_dur_max,
            self.sampling_rate,
            self.down_factors,
            self.sil_phones,
            self.add_blank,
            self.seed,
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.trainset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            collate_fn=TextCodesBatchCollate(
                device=self.device,
                prompt_max_len=self.prompt_dur_max * self.sampling_rate // np.prod(self.down_factors),
                prompt_reduced_factor=self.prompt_reduced_factor,
                vocab_size=self.vocab_size,
            ),
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.validset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            collate_fn=TextCodesBatchCollate(
                device=self.device,
                prompt_max_len=self.prompt_dur_max * self.sampling_rate // np.prod(self.down_factors),
                prompt_reduced_factor=self.prompt_reduced_factor,
                vocab_size=self.vocab_size,
            ),
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass  # pylint: disable=unnecessary-pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass  # pylint: disable=unnecessary-pass


class TextCodesDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root,
        manifest,
        cleaners,
        dur_min=0.3,
        dur_max=15,
        n_words_min=3,
        prompt_dur_max=3,
        sampling_rate=16000,
        down_factors=None,
        sil_phones=None,
        add_blank=True,
        seed=None,
    ):
        self.data_root = data_root
        self.manifest = manifest
        self.cleaners = cleaners
        self.dur_min = dur_min
        self.dur_max = dur_max
        self.prompt_dur_max = prompt_dur_max
        self.sampling_rate = sampling_rate
        self.sil_phones = sil_phones
        self.add_blank = add_blank

        if down_factors is None:
            self.down_factors = [2, 4, 5, 5]
        else:
            self.down_factors = down_factors
        self.down_factor = np.prod(self.down_factors)
        
        if sil_phones is None:
            self.sil_phones = ["sil", "sp", "spn", ""]
        else:
            self.sil_phones = sil_phones
            
        samples, filters, dur_total = [], [], 0
        with open(os.path.join(self.data_root, self.manifest), 'r', encoding='utf-8') as manifest:
            for line in manifest:
                sample = line.replace('\n', '')
                duration = float(sample.split('|')[1])
                n_words = len(sample.split('|')[2].split(' '))
                
                if duration < self.dur_min or duration > self.dur_max or n_words < n_words_min:
                    filters.append(sample)
                    continue
                samples.append(sample)        
                dur_total += duration
                
        dur_total = round(dur_total / 3600, 3)
        self.samples = samples

        print('+-'*50)
        print(f'>>>\t {self.manifest}: {dur_total} hours')
        print(f'>>>\t Valid utterances: {len(self.samples)}')
        print(f'>>>\t Filtered utterances: {len(filters)}')
        print('+-'*50)
                
        random.seed(seed)
        random.shuffle(self.samples)

    def get_datapoint(self, sample):
        (
            filename, 
            dur_in_sec, 
            transcript,
            style_prompt,
            textgrid_path,
            tgt_codes_path,
            cond_codes_path,
        ) = tuple(sample.split('|'))
        
        textgrid = tgt.io.read_textgrid(textgrid_path, include_empty_intervals=True)

        tgt_codes = json.load(open(tgt_codes_path))['quantizers']
        tgt_codes = torch.stack([torch.IntTensor(quantizer) for quantizer in tgt_codes])

        phones, durations, tgt_codes, _, _ = self.get_alignment(
            textgrid.get_tier_by_name("phones"), 
            tgt_codes,
        )
        durations = torch.IntTensor(durations)
        phonemes = torch.IntTensor(text_to_sequence('{' + ' '.join(phones) + '}', self.cleaners))
        
        return {
            "x": phonemes, 
            "y": tgt_codes,
            "dur": durations,
            "style": style_prompt,
            "filename": filename, 
            "text_clean": phones,
            "text_raw": transcript,
        }
        
    def get_cond_segment(self, textgrid_tier, codes):
        dur_reserved = self.prompt_dur_max * self.sampling_rate // self.down_factor
        for t in textgrid_tier._objects:
            s, e, _ = t.start_time, t.end_time, t.text
            start_idx = int(s * self.sampling_rate // self.down_factor)
            end_idx = int(e * self.sampling_rate // self.down_factor)
            dur_reserved -= end_idx - start_idx
            if dur_reserved <= 0:
                codes = codes[:,:end_idx+1]
                break
        return codes
    
    def get_alignment(self, textgrid_tier, codes):

        phones, durations = [], []
        start_time, end_time, end_idx = 0, 0, 0
        
        for t in textgrid_tier._objects:
            s, e, p = t.start_time, t.end_time, t.text

            # Trim leading silences
            if phones == []:
                if p in self.sil_phones:
                    continue
                else:
                    start_time = s  

            if p not in self.sil_phones:
                # For ordinary phones
                phones.append(p)
                end_time = e
                end_idx = len(phones)
            else:
                # For silent phones
                p = 'sp' if p == '' else p
                phones.append(p)
                
            start_code = s * self.sampling_rate // self.down_factor
            end_code = e * self.sampling_rate // self.down_factor

            durations.append(end_code - start_code)

        # Trim tailing silences
        phones = phones[:end_idx]
        durations = durations[:end_idx]
        
        start_code_trim = start_time * self.sampling_rate // self.down_factor
        end_code_trim = end_time * self.sampling_rate // self.down_factor
        codes = codes[:, int(start_code_trim):int(end_code_trim)]
        
        return phones, durations, codes, start_time, end_time

    def __getitem__(self, index):
        datapoint = self.get_datapoint(self.samples[index])
        return datapoint

    def __len__(self):
        return len(self.samples)


class TextCodesBatchCollate:
    def __init__(
        self, 
        device,
        prompt_max_len=720,
        prompt_reduced_factor=0.8,
        vocab_size=1024,
        ):
        
        self.device = device
        self.vocab_size = vocab_size
        self.prompt_max_len = prompt_max_len
        self.prompt_reduced_factor = prompt_reduced_factor
        
    def _process_acoustic_prompt(self, prompts):
        max_len = min([prompt.size(1) for prompt in prompts] + [self.prompt_max_len])
        max_len_reduced = int(self.prompt_reduced_factor * max_len)
        
        prompt_segments = []
        for prompt in prompts:
            start_idx = random.randint(0, prompt.size(1) - max_len_reduced)
            end_idx = start_idx + max_len_reduced
            prompt_segments.append(prompt[:,start_idx:end_idx])
            
        prompts = torch.stack(prompt_segments)
        # mask content quantizer
        prompts[:,1:3,:] = self.vocab_size
        # add eos
        bs, qs, _ = prompts.shape
        eos = torch.zeros((bs, qs, 1), dtype=prompts.dtype) + self.vocab_size
        prompts = torch.cat([prompts, eos], dim=-1)
        return prompts
    
    def __call__(self, batch):
        B = len(batch)
        x_max_len = max([item["x"].shape[-1] for item in batch])
        y_max_len = max([item["y"].shape[-1] for item in batch])
        n_codes = batch[0]["y"].shape[-2]

        x = torch.zeros(
            (B, x_max_len), 
            dtype=torch.long, device=self.device
        )
        y = torch.zeros(
            (B, n_codes, y_max_len), 
            dtype=torch.long, device=self.device
        ) + self.vocab_size
        durs = torch.zeros(
            (B, x_max_len), 
            dtype=torch.long, device=self.device
        )

        prompts, style, x_len, y_len, filenames = [], [], [], [], []
        
        for i, item in enumerate(batch):
            x_i, y_i, dur_i = item["x"], item["y"], item["dur"]
            x[i, : x_i.shape[-1]] = x_i
            y[i, :, : y_i.shape[-1]] = y_i
            durs[i, : dur_i.shape[-1]] = dur_i
            prompts.append(y_i)
            filenames.append(item['filename'])

            y_len.append(y_i.shape[-1])
            x_len.append(x_i.shape[-1])
            style.append(item["style"])

        x_len = torch.tensor(x_len, dtype=torch.int, device=self.device)
        y_len = torch.tensor(y_len, dtype=torch.int, device=self.device)
        
        prompts = self._process_acoustic_prompt(prompts)
        del batch

        return (
            x, 
            x_len, 
            y, 
            y_len, 
            durs,
            prompts,
            filenames
        )