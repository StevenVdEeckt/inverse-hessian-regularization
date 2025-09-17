from espnet2.tasks.asr import ASRTask
import torch
import logging
from torch import nn
from decimal import Decimal
import os
import argparse
import numpy as np

parser = argparse.ArgumentParser(description="Process text files to add punctuation, capitalize sentences, and handle special tokens.")
parser.add_argument("model", type=str, help="Directory containing the text files to process.")
parser.add_argument("--alpha", type=float, default=1.0, help="Weight for the averaging")
args = parser.parse_args()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# load current model
asr_train_config = f"{args.model}/config.yaml"
asr_model_file   = f"{args.model}/valid.acc.ave.pth"
device = "cpu"

asr_model, asr_train_args = ASRTask.build_model_from_file(
        asr_train_config, asr_model_file, device
)

# load previous model
init_model = torch.load(f"{args.model}/initial_model.pth", map_location='cpu')

# to store new model
new_model = {}

# apply averaging
for name, param in asr_model.state_dict().items():
    delta_W = param.cpu() - init_model[name] 
    gated_update = args.alpha * delta_W
    new_model[name] = init_model[name] + gated_update

torch.save(new_model, f"{args.model}/fta_model.pth")
logging.info(f"Done!")
