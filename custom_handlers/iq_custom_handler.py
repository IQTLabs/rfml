import json
import numpy as np
import os
import time
import torch

from collections import defaultdict
from ts.torch_handler.base_handler import BaseHandler

try:
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
except ImportError:
    XLA_AVAILABLE = False

# test with:
# $ curl --header "Content-Type:application/octet-stream" --data-binary @/path/to/datablobfile http://localhost:8080/predictions/torchsig_model
#
# create MAR with:
#
# $ torch-model-archiver --force --model-name torchsig_model --version 1.0 --serialized-file checkpoint-v16_torchscript.pt --handler custom_handler.py


class TorchsigHandler(BaseHandler):
    """
    A custom model handler implementation.
    """

    def __init__(self):
        self.initialized = False
        self.explain = False
        # self.target = 0
        self.handler_name = "TorchsigHandler"

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"{self.handler_name}: using CUDA")
        elif XLA_AVAILABLE:
            self.device = xm.xla_device()
            print(f"{self.handler_name}: using XLA")
        else:
            self.device = torch.device("cpu")
            print(f"{self.handler_name}: using CPU")

        self.model_config = {}
        if os.path.exists("model_configs.json"):
            with open("model_config.json", "r") as f:
                self.model_config = json.load(f)

        self.manifest = context.manifest
        model_dir = context.system_properties.get("model_dir")
        self.model_pt_path = None
        if "serializedFile" in self.manifest["model"]:
            serialized_file = self.manifest["model"]["serializedFile"]
            self.model_pt_path = os.path.join(model_dir, serialized_file)
        self.model = self._load_torchscript_model(self.model_pt_path)
        self.initialized = True
        self.avg_size = 0
        self.avg_db_historical = 0
        self.max_db = None

    def add_to_avg(self, value):
        self.avg_db_historical = (self.avg_db_historical * self.avg_size + value) / (
            self.avg_size + 1
        )
        self.avg_size += 1

    def _load_torchscript_model(self, model_pt_path):
        """Loads the PyTorch model and returns the NN model object.

        Args:
                model_pt_path (str): denotes the path of the model file.

        Returns:
                (NN Model Object) : Loads the model object.
        """
        # TODO: remove this method if https://github.com/pytorch/text/issues/1793 gets resolved

        # Load JIT model
        model = torch.jit.load(model_pt_path)

        # Load model weights
        # model_checkpoint = torch.load(model_pt_path)
        # num_classes = len(model_checkpoint["classifier.bias"])
        # model = efficientnet_b4(
        # 			num_classes=num_classes
        # )
        # model.load_state_dict(model_checkpoint)

        model = model.to(self.device)
        return model

    def handle(self, data, context):
        """Entry point for default handler. It takes the data from the input request and returns
                the predicted outcome for the input.

        Args:
                data (list): The input data that needs to be made a prediction request on.
                context (Context): It is a JSON Object containing information pertaining to
                                                        the model artifacts parameters.

        Returns:
                list : Returns a list of dictionary with the predicted response.
        """

        # It can be used for pre or post processing if needed as additional request
        # information is available in context
        start_time = time.time()

        self.context = context
        metrics = self.context.metrics

        is_profiler_enabled = os.environ.get("ENABLE_TORCH_PROFILER", None)
        if is_profiler_enabled:
            if PROFILER_AVAILABLE:
                if self.manifest is None:
                    # profiler will use to get the model name
                    self.manifest = context.manifest
                output, _ = self._infer_with_profiler(data=data)
            else:
                raise RuntimeError(
                    "Profiler is enabled but current version of torch does not support."
                    "Install torch>=1.8.1 to use profiler."
                )
        else:
            if self._is_describe():
                output = [self.describe_handle()]
            elif self.gate(data):
                print(f"\n NO SIGNAL \n")
                output = [{"No signal": None}]
            else:
                data_preprocess = self.preprocess(data)

                if not self._is_explain():
                    output = self.inference(data_preprocess)
                    output = self.postprocess(output)
                else:
                    output = self.explain_handle(data_preprocess, data)

        stop_time = time.time()
        metrics.add_time(
            "HandlerTime", round((stop_time - start_time) * 1000, 2), None, "ms"
        )
        return output

    def gate(self, data):
        """
        Gates input data for inference
        :param data: list of raw requests, should match batch size
        :return: bool True if gate input, False if for inference
        """
        try:
            body = data[0]["body"]
        except (KeyError, IndexError) as exc:
            print("could not parse body from request: ", data)
            raise ValueError from exc

        data = torch.tensor(np.frombuffer(body, dtype=np.complex64), dtype=torch.cfloat)

        avg_pwr = 10 * torch.log10(torch.mean(torch.abs(data) ** 2))
        if self.max_db is None or self.max_db < avg_pwr:
            self.max_db = avg_pwr

        self.add_to_avg(avg_pwr)

        print("\n=====================================\n")
        print("\n=====================================\n")
        print(f"\n{data=}\n")
        print(f"\n{avg_pwr=}, \n{self.max_db=}, \n{self.avg_db_historical=}\n")
        print(f"\n{torch.min(torch.abs(data)**2)=}, {torch.max(torch.abs(data)**2)=}\n")
        print("\n=====================================\n")
        print("\n=====================================\n")

        if avg_pwr > (self.max_db + self.avg_db_historical) / 2:
            return False

        return True

    def preprocess(self, data):
        """
        Transform raw input into model input data.
        :param data: list of raw requests, should match batch size
        :return: list of preprocessed model input data
        """
        try:
            body = data[0]["body"]
        except (KeyError, IndexError) as exc:
            print("could not parse body from request: ", data)
            raise ValueError from exc

        data = torch.tensor(np.frombuffer(body, dtype=np.complex64), dtype=torch.cfloat)
        print("\n=====================================\n")
        print(f"\n{data=}\n")
        print(f"\n{torch.min(torch.abs(data)**2)=}, {torch.max(torch.abs(data)**2)=}\n")
        avg_pwr = torch.mean(torch.abs(data) ** 2)
        avg_pwr_db = 10 * torch.log10(avg_pwr)
        print(f"\n{avg_pwr=}\n")
        print(f"\n{avg_pwr_db=}\n")
        data = data * 1 / torch.norm(data, float("inf"), keepdim=True)
        data = torch.view_as_real(data)
        data = torch.movedim(data, -1, 0).unsqueeze(0)
        # data should be of size (N, 2, n_samples)

        data = data.to(self.device)
        print("\n=====================================\n")
        return data

    def inference(self, model_input):
        """
        Internal inference methods
        :param model_input: transformed model input data
        :return: list of inference output in NDArray
        """
        # Do some inference call to engine here and return output
        model_output = self.model.forward(model_input)
        return model_output

    def postprocess(self, inference_output):
        """
        Return inference result.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        confidences, class_indexes = torch.max(inference_output.data, 1)
        results = {
            str(class_index): [{"confidence": confidence}]
            for class_index, confidence in zip(
                class_indexes.tolist(), confidences.tolist()
            )
        }
        print(f"\n{inference_output=}\n{results=}\n")
        return [results]
