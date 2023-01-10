import torch
import unittest

from mqbench.prepare_by_platform import prepare_by_platform, BackendType
from mqbench.convert_deploy import convert_deploy
from mqbench.utils.state import enable_calibration, enable_quantization

from resnet import resnet18

model_to_quantize = torch.hub.load(GITHUB_RES, 'resnet18', pretrained=False)
dummy_input = torch.randn(2, 3, 224, 224, device='cpu')
model_to_quantize.train()
extra_qconfig_dict = {
    'w_observer': 'MinMaxObserver',
    'a_observer': 'EMAMinMaxObserver',
    'w_fakequantize': 'FixedFakeQuantize',
    'a_fakequantize': 'FixedFakeQuantize',
}
prepare_custom_config_dict = {'extra_qconfig_dict': extra_qconfig_dict}
model_prepared = prepare_by_platform(model_to_quantize, BackendType.Tensorrt, prepare_custom_config_dict)
enable_calibration(model_prepared)
model_prepared(dummy_input)
enable_quantization(model_prepared)
loss = model_prepared(dummy_input).sum()
loss.backward()
model_prepared.eval()
convert_deploy(model_prepared, BackendType.Tensorrt, {'x': [1, 3, 224, 224]}, model_name='resnet18_fixed.onnx')
