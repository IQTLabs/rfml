import torch
from collections import OrderedDict
from torchsig.models.iq_models.efficientnet.efficientnet import efficientnet_b0
import os
import argparse
import subprocess


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, help="Name of the model", required=True
    )
    parser.add_argument(
        "--checkpoint", type=str, help="Path to the model checkpoint", required=True
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["export, convert"],
        default="export",
        help="Whether to convert model to torchserve/torchscript or export to MAR. 'export' will automatically convert the checkpoint and export to MAR. (default: %(default)s)",
    )
    parser.add_argument(
        "--custom_handler",
        type=str,
        default="custom_handlers/iq_custom_handler.py",
        help="Custom handler to use when exporting to MAR. Only used if --mode='export'. (default: %(default)s)",
    )
    parser.add_argument(
        "--export_path",
        type=str,
        default="models/",
        help="Path to export MAR file to. Only used if --mode='export'. (default: %(default)s)",
    )

    return parser


def convert_model(model_name, checkpoint):

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

    model = efficientnet_b0(num_classes=num_classes)

    model.load_state_dict(model_weights)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    x = torch.randn(1, 2, 1024).to(device)
    model(x)

    model.eval()
    print(f"Model Test: {model(x)}")
    jit_net = torch.jit.trace(model, x)
    torchscript_file = f"weights/{model_name}_torchscript.pt"
    jit_net.save(torchscript_file)  # Save

    print(f"Saved torchscript version of model to: {torchscript_file}")

    return torchscript_file


def export_model(model_name, torchscript_file, custom_handler, export_path):

    torch_model_archiver_args = [
        "torch-model-archiver",
        "--force",
        "--model-name",
        model_name,
        "--version",
        "1.0",
        "--serialized-file",
        torchscript_file,
        "--handler",
        custom_handler,
        "--export-path",
        export_path,
        "-r",
        "custom_handlers/requirements.txt",
    ]

    subprocess.run(torch_model_archiver_args)


if __name__ == "__main__":

    args = argument_parser().parse_args()

    torchscript_file = convert_model(args.model_name, args.checkpoint)

    if args.mode == "export":
        export_model(
            args.model_name, torchscript_file, args.custom_handler, args.export_path
        )
