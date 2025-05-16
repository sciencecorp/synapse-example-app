#!/usr/bin/env python3
import warnings
warnings.simplefilter("ignore", DeprecationWarning)
import os
import time
import copy
import argparse
from typing import Optional, List
import torch
from omegaconf import OmegaConf
import wave
import glob
import threading
import queue
from queue import Queue
import numpy as np
import sounddevice as sd  # continuous low-latency playback
import audioop

from model import GRUDecoder
from synapse.client.taps import Tap
from synapse.api.datatype_pb2 import Tensor

from rich.console import Console, Group
from rich.live import Live
from rich.status import Status
from rich.text import Text

# -----------------------------------------------------------------------------
# Mapping from model output indices to phoneme labels. This mirrors the
# LOGIT_PHONE_DEF list used during model training.
# Index 0 is the CTC "blank" token and index 1 is a silence marker (SIL).
# -----------------------------------------------------------------------------
LOGIT_PHONE_DEF = [
    'BLANK', 'SIL',
    'AA', 'AE', 'AH', 'AO', 'AW',
    'AY', 'B',  'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G',
    '_H', 'IH', 'IY', 'J', 'K',
    'L', 'M', 'N', 'NX', 'OW',
    'OY', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UW', 'V',
    'W', 'Y', 'Z', 'ZH'
]


def decode_logits_to_phonemes(logits):
    """Greedy CTC-style decoding of model logits to a phoneme sequence.

    Args:
        logits (torch.Tensor): shape (1, T, C) where C == len(LOGIT_PHONE_DEF)

    Returns:
        List[str]: Collapsed phoneme sequence with BLANK tokens removed.
    """
    # Convert logits → class index for each timestep.
    pred_indices = logits.argmax(dim=-1).squeeze(0).cpu().tolist()

    # Collapse repeats and drop BLANK (index 0).
    phoneme_seq = []
    prev_idx = None
    for idx in pred_indices:
        if idx == prev_idx:
            continue  # skip repeats
        prev_idx = idx
        if idx == 0:  # BLANK
            continue
        phoneme_seq.append(LOGIT_PHONE_DEF[idx])
    return phoneme_seq

class SynappClient(object):
    def __init__(self, model, device, tap):
        self.model = model
        self.device = device
        self.tap = tap

        self.new_feature_period = 4
        self.feature_window_len = 14
        self.n_day_layers = len(self.model.day_weights)
        self.tot_group_time = 0
        self.max_group_time = 0
        self.min_group_time = 1000000

        # ------------------------------------------------------------------
        # Real-time audio playback setup (single OutputStream)
        # ------------------------------------------------------------------
        self.playback_queue: Queue[str] = Queue(maxsize=512)
        self.phoneme_frames, self.silence_frames, self.sample_rate = self._prepare_phoneme_frames()

        # A single PortAudio stream keeps the audio device open, eliminating
        # repeated buffer allocation/free that caused Core Audio crashes.
        self.stream = sd.OutputStream(samplerate=self.sample_rate,
                                      channels=1,
                                      dtype='int16',
                                      blocksize=0,
                                      latency='low')
        self.stream.start()

        # Background thread drains queue and writes frames.
        threading.Thread(target=self._audio_playback_loop, daemon=True).start()

    # ----------------------------------------------------------------------
    # Helper utilities for loading and playing short phoneme clips
    # ----------------------------------------------------------------------
    def _generate_silence(self, sample_rate: int = 16000, duration_sec: float = 0.08,
                           channels: int = 1, sampwidth: int = 2):
        n_frames = int(sample_rate * duration_sec)
        return (bytes(n_frames * channels * sampwidth), channels, sampwidth, sample_rate)

    def _prepare_phoneme_frames(self, wav_root: str = "phoneme_wavs", duration_sec: float = 0.08):
        """Load ≤duration_sec audio snippets for every phoneme label.

        Returns
        -------
        tuple(dict, tuple):
            1) mapping phoneme -> (audio_bytes, n_channels, sampwidth, sample_rate)
            2) fallback silence tuple of same structure
        """
        # Build silence fallback (80 ms of silence) once.
        silence_bytes, default_ch, default_sw, default_sr = self._generate_silence()
        silence_np = np.frombuffer(silence_bytes, dtype=np.int16)

        mapping: dict[str, np.ndarray] = {}

        # Find all wavs in directory once.
        for wav_path in glob.glob(os.path.join(wav_root, "*.wav")):
            key = os.path.splitext(os.path.basename(wav_path))[0].upper()

            try:
                wf = wave.open(wav_path, 'rb')
            except wave.Error:
                continue  # skip unreadable files

            sr = wf.getframerate()
            nch = wf.getnchannels()
            sw = wf.getsampwidth()

            # simpleaudio supports 1 or 2 channels and 1- or 2-byte samples.
            # Convert anything outside that range to 16-bit mono.
            raw_frames = wf.readframes(int(sr * duration_sec))

            # Ensure 16-bit mono for PortAudio
            if sw not in (1, 2) or nch not in (1, 2):
                # Convert to 16-bit mono using audioop. This avoids rare 24-bit or
                # 32-bit encodings that can make PortAudio segfault.
                try:
                    # First, ensure we are working with 16-bit little-endian
                    raw_frames = audioop.lin2lin(raw_frames, sw, 2)
                    sw = 2
                except audioop.error:
                    # Fallback to our pre-built silence WaveObject.
                    mapping[key] = silence_np
                    continue

                # Down-mix to mono if necessary
                if nch > 1:
                    raw_frames = audioop.tomono(raw_frames, 2, 0.5, 0.5)
                nch = 1

                # Truncate/pad to exact duration again after conversion
                max_frames = int(sr * duration_sec)
                raw_frames = raw_frames[:max_frames * nch * sw].ljust(max_frames * nch * sw, b"\x00")
                mapping[key] = np.frombuffer(raw_frames, dtype=np.int16)
                continue

            # For supported 8-/16-bit data keep original but rename for later math
            frames = raw_frames
            max_frames = int(sr * duration_sec)
            wf.close()

            need = max_frames * nch * sw - len(frames)
            if need > 0:
                frames += bytes(need)
            else:
                frames = frames[:max_frames * nch * sw]

            mapping[key] = np.frombuffer(frames, dtype=np.int16)

        # Provide at least SIL and BLANK as silence
        mapping.setdefault('SIL', silence_np)
        mapping.setdefault('BLANK', silence_np)

        return mapping, silence_np, default_sr

    def _audio_playback_loop(self):
        """Continuously play queued phoneme audio snippets."""
        while True:
            phoneme = self.playback_queue.get()
            frames = self.phoneme_frames.get(phoneme.upper(), self.silence_frames)
            # PortAudio expects shape (N,) or (N, 1) for mono.
            self.stream.write(frames)
            self.playback_queue.task_done()

    @torch.no_grad()
    @torch.autocast(device_type = "cuda", enabled = False, dtype = torch.bfloat16)
    def run(self):
        console = Console()
        
        self.model.eval()
        group_start_time = time.time()
        tot_groups = 0
        phoneme_history = []
        
        # Function to get stats text
        def get_stats_text():
            if tot_groups == 0:
                avg_time = 0
            else:
                avg_time = self.tot_group_time / tot_groups
            return f"Samples: {tot_groups} | Avg: {avg_time:.4f}s | Min: {self.min_group_time:.4f}s | Max: {self.max_group_time:.4f}s"
        
        # Create a live display with spinner
        with Live(console=console, refresh_per_second=10) as live:
            title_text = Text.from_markup("[bold cyan]Synapse Neural Decoding[/bold cyan]")
            
            while True:
                current_renderables = [title_text]
                stats_str = get_stats_text()
                stats_display_text = Text.from_markup(stats_str)

                newbytes : Optional[bytes] = self.tap.read()
                if newbytes is None:
                    current_status_obj = Status("Waiting for data...", spinner="dots", spinner_style="blue")
                    current_renderables.extend([current_status_obj, stats_display_text])
                    live.update(Group(*current_renderables))
                    continue
                    
                syn_tensor = Tensor()
                syn_tensor.ParseFromString(newbytes)
                tensor_folded = torch.frombuffer(bytearray(syn_tensor.data), dtype=torch.float32)
                tensor = tensor_folded.view(syn_tensor.shape[0], syn_tensor.shape[1])

                in_features = tensor
                logits = self.model(x = in_features.unsqueeze(0).to(self.device), day_idx = torch.tensor([self.n_day_layers - 1]).to(self.device))

                # Update timing stats
                group_process_time = time.time() - group_start_time
                tot_groups += 1
                if group_process_time > self.max_group_time:
                    self.max_group_time = group_process_time
                if group_process_time < self.min_group_time:
                    self.min_group_time = group_process_time
                self.tot_group_time += group_process_time
                group_start_time = time.time()

                # Decode logits to phonemes
                phoneme_sequence = decode_logits_to_phonemes(logits)
                
                # Queue phonemes for playback
                for ph_item in phoneme_sequence:
                    try:
                        self.playback_queue.put_nowait(ph_item)
                    except queue.Full:
                        try:
                            _ = self.playback_queue.get_nowait()
                        except queue.Empty:
                            pass
                        self.playback_queue.put_nowait(ph_item)
                
                # Update display
                current_status_obj = Status("Processing audio", spinner="dots", spinner_style="green")
                current_renderables.extend([current_status_obj, stats_display_text])
                
                # --- Phoneme Display Section ---
                current_renderables.append(Text()) # Blank line for spacing
                current_renderables.append(Text.from_markup("[yellow bold]Phonemes:[/yellow bold]"))

                if not phoneme_sequence:
                    # Current segment has no phonemes
                    current_renderables.append(Text.from_markup("[i]No phonemes detected in this segment.[/i]"))
                else:
                    # Current segment has phonemes, update history
                    phoneme_text_str = " → ".join(phoneme_sequence)
                    phoneme_history.append(phoneme_text_str)
                    if len(phoneme_history) > 3:
                        phoneme_history = phoneme_history[-3:]
                
                # Always display phoneme history if it's not empty
                if phoneme_history:
                    for idx, ph_hist_item_str in enumerate(phoneme_history):
                        history_item_text = Text(f"{idx+1}. {ph_hist_item_str}")
                        current_renderables.append(history_item_text)
                # If phoneme_history is empty and current segment was also empty, 
                # the "No phonemes..." message added above will be the only item under the title.
                live.update(Group(*current_renderables))

def init_model(model_path: str, model_compiled: bool = False) -> GRUDecoder:
    console = Console()
    
    with console.status("[bold blue]Loading model...", spinner="dots") as status:
        # Load configuration and initialize model
        model_args = OmegaConf.load(os.path.join(model_path, 'checkpoint/args.yaml'))
        neural_dim = model_args['model']['n_input_features']

        model = GRUDecoder(
            neural_dim = neural_dim,
            n_units = model_args['model']['n_units'],
            n_layers = model_args['model']['n_layers'],
            n_classes = model_args['dataset']['n_classes'],
            patch_size = model_args['model']['patch_size'],
            patch_stride = model_args['model']['patch_stride'],
            input_dropout = model_args['model']['input_network']['input_layer_dropout'],
            rnn_dropout = model_args['model']['rnn_dropout'],
            n_days= len(model_args['dataset']['sessions'])
        )

        # Load and process weights
        checkpoint = torch.load(os.path.join(model_path, 'checkpoint/best_checkpoint'), weights_only=False, map_location='cpu')

        # Process and load model state dict
        for key in list(checkpoint['model_state_dict'].keys()):
            checkpoint['model_state_dict'][key.replace("module.", "")] = checkpoint['model_state_dict'].pop(key)
            if not model_compiled:
                checkpoint['model_state_dict'][key.replace("_orig_mod.", "")] = checkpoint['model_state_dict'].pop(key)
        
        model.load_state_dict(checkpoint['model_state_dict'])

    console.print("[bold blue]Model loaded successfully ✓[/bold blue]")
    return model

def parse_args():
    parser = argparse.ArgumentParser(description="Synapse neural decoding client")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the model checkpoint directory")
    parser.add_argument("--device-ip", type=str, required=True,
                        help="IP address of the Synapse device")
    parser.add_argument("--tap-name", type=str, required=True,
                        help="Name of the tap to connect to")
    args = parser.parse_args()
    
    # Validate model path
    if not os.path.exists(args.model_path):
        parser.error(f"Model path '{args.model_path}' does not exist")
        
    return args

def main():
    console = Console()
    args = parse_args()
    
    model = init_model(model_path=args.model_path)
    dev_tap = Tap(args.device_ip)
    dev_tap.connect(args.tap_name)
    console.print(f"[bold green]Connected to device tap '{args.tap_name}' at {args.device_ip}[/bold green]")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    client_app = SynappClient(model, device, dev_tap)

    try:
        client_app.run()
    except KeyboardInterrupt:
        print("Stopping the client...")

if __name__ == "__main__":
    main()