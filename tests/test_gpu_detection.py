import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from transcriptex.asr import AsrConfig, _auto_optimize_config
from transcriptex.dependency_manager import cuda_available, nvidia_gpu_info


class GPUDetectionTests(unittest.TestCase):
    def test_ct2_unavailable_does_not_use_torch_as_a_false_positive(self):
        with patch('transcriptex.dependency_manager._has_nvidia_gpu',return_value=True), patch('transcriptex.dependency_manager.ensure_nvidia_cuda_runtime'), patch.dict(sys.modules,{'ctranslate2':SimpleNamespace(get_cuda_device_count=lambda:0),'torch':SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:True))}):
            self.assertFalse(cuda_available())

    def test_vram_probe_parses_nvidia_smi(self):
        with patch('transcriptex.dependency_manager.subprocess.run',return_value=SimpleNamespace(stdout='NVIDIA RTX 4060, 8188\n')):
            self.assertEqual(nvidia_gpu_info(),('NVIDIA RTX 4060',8188))

    def test_unknown_vram_preserves_requested_settings(self):
        config=AsrConfig(channel_dir=Path('.'),device='cuda',compute_type='float16',batch_size=16)
        with patch('transcriptex.dependency_manager.cuda_available',return_value=True), patch('transcriptex.dependency_manager.nvidia_gpu_info',return_value=None):
            self.assertIs(_auto_optimize_config(config,quiet=True),config)

    def test_known_vram_applies_existing_small_gpu_limits(self):
        config=AsrConfig(channel_dir=Path('.'),device='cuda',compute_type='float16',batch_size=16)
        with patch('transcriptex.dependency_manager.cuda_available',return_value=True), patch('transcriptex.dependency_manager.nvidia_gpu_info',return_value=('Small GPU',4096)):
            tuned=_auto_optimize_config(config,quiet=True)
        self.assertEqual((tuned.compute_type,tuned.batch_size),('int8',4))
        self.assertEqual(config.batch_size,16)


if __name__=='__main__':
    unittest.main()
