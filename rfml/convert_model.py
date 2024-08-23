import torch
from collections import OrderedDict
from torchsig.models.iq_models.efficientnet.efficientnet import efficientnet_b4
import os
import argparse

model_name = "drone_detection"
checkpoint = "/home/iqt/lberndt/rfml-dev-1/rfml-dev/lightning_logs/version_5/checkpoints/experiment_logs/experiment_1/iq_checkpoints/checkpoint.ckpt"

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, help="Name of the model", required=True)
parser.add_argument(
    "--checkpoint", type=str, help="Path to the model checkpoint", required=True
)
args = parser.parse_args()

model_name = args.model_name
checkpoint = args.checkpoint

model_checkpoint = torch.load(checkpoint)
print(f"Loaded model checkpoint from {checkpoint}")
model_weights = model_checkpoint["state_dict"]
model_weights = OrderedDict(
    (k.removeprefix("mdl."), v) for k, v in model_weights.items()
)
num_classes = len(model_weights["classifier.bias"])
print(f"Model has {num_classes} classes")
if not os.path.exists("weights"):
    os.makedirs("weights")

torch.save(model_weights, f"weights/{model_name}_torchserve.pt")
print(f"Saved model weights to weights/{model_name}_torchserve.pt")

model = efficientnet_b4(num_classes=num_classes)

model.load_state_dict(model_weights)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

x = torch.randn(1, 2, 1024).to(device)
model(x)

model.eval()
print(f"Model Test: {model(x)}")
jit_net = torch.jit.trace(model, x)
jit_net.save(f"weights/{model_name}_torchscript.pt")  # Save

print(f"Saved torchscript version of model to: weights/{model_name}_torchscript.pt")
