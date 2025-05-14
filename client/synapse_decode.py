#!/usr/bin/env python3
import os
import time
import copy
from typing import Optional, List
import torch
from omegaconf import OmegaConf

from model import GRUDecoder
from synapse.client.taps import Tap
from synapse.api.datatype_pb2 import Tensor

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

    @torch.no_grad()
    @torch.autocast(device_type = "cuda", enabled = False, dtype = torch.bfloat16)
    def run(self):
        self.model.eval()

        group_start_time = time.time()
        tot_groups = 0
        while True:
            newbytes : Optional[bytes] = self.tap.read()
            if newbytes is None:
                continue
            syn_tensor = Tensor()
            syn_tensor.ParseFromString(newbytes)
            tensor_folded = torch.frombuffer(bytearray(syn_tensor.data), dtype=torch.float32)
            tensor = tensor_folded.view(syn_tensor.shape[0], syn_tensor.shape[1])


            in_features = tensor
            start = time.time()
            logits = self.model(x = in_features.unsqueeze(0).to(self.device), day_idx = torch.tensor([self.n_day_layers - 1]).to(self.device)) 
            end = time.time()

            group_process_time = end - group_start_time
            tot_groups += 1
            if group_process_time > self.max_group_time:
                self.max_group_time = group_process_time
            if group_process_time < self.min_group_time:
                self.min_group_time = group_process_time

            print("Time taken for model inference: ", end - start)
            print("Time to gather and process features: ", group_process_time)
            self.tot_group_time += group_process_time
            print("Avg time per group (4 bins): ", self.tot_group_time / tot_groups)
            print("Max time per group (4 bins): ", self.max_group_time)
            print("Min time per group (4 bins): ", self.min_group_time)
            group_start_time = time.time()
            print("Logits:", logits.shape, logits)


def init_model(model_path: str, model_compiled: bool = False) -> GRUDecoder:

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

    # Load the model weights
    checkpoint = torch.load(os.path.join(model_path, 'checkpoint/best_checkpoint'), weights_only=False, map_location='cpu')

    # rename model keys to not start with "module." (happens if model was saved with DataParallel)
    for key in list(checkpoint['model_state_dict'].keys()):
        checkpoint['model_state_dict'][key.replace("module.", "")] = checkpoint['model_state_dict'].pop(key)
        if not model_compiled:
            # remove "orig_mod." from keys if they exist (happens if model was saved with torch.compile)
            checkpoint['model_state_dict'][key.replace("_orig_mod.", "")] = checkpoint['model_state_dict'].pop(key)
    # Load the model state dict
    model.load_state_dict(checkpoint['model_state_dict'])

    print("Model loaded successfully.")
    return model

def main():
    # device_ip = "10.40.63.208"
    device_ip = "192.168.0.165"
    tap_name = "features_out"
    model_path = "./ckpts/rnn_model"


    model = init_model(model_path=model_path)  # Replace with your model initialization
    dev_tap = Tap(device_ip)
    dev_tap.connect(tap_name)
    print("Connected to device tap.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    client_app = SynappClient(model, device, dev_tap)

    try:
        client_app.run()
    except KeyboardInterrupt:
        print("Stopping the client...")

if __name__ == "__main__":
    main()