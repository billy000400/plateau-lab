"""Selection-policy tests with simulated GPU APIs; no CUDA hardware required."""
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
from hardware import Hardware, is_out_of_memory

GB = 1024 ** 3


class HardwareChecks(unittest.TestCase):
    def devices(self, free=(2 * GB, 12 * GB), mps=False, failed=(), hip=None):
        stack = ExitStack()
        self.addCleanup(stack.close)
        def probe(device):
            if device in failed:
                raise RuntimeError('device kernel unavailable')
        stack.enter_context(patch('hardware.probe_device', side_effect=probe))
        stack.enter_context(patch('torch.cuda.is_available', return_value=bool(free)))
        stack.enter_context(patch('torch.cuda.device_count', return_value=len(free)))
        stack.enter_context(patch('torch.cuda.get_device_name', side_effect=lambda i: f'Test GPU {i}'))
        stack.enter_context(patch('torch.cuda.mem_get_info', side_effect=lambda i: (free[i if isinstance(i, int) else i.index], 24 * GB)))
        stack.enter_context(patch('torch.backends.mps.is_available', return_value=mps))
        stack.enter_context(patch('torch.version.hip', hip))

    def test_auto_prefers_gpu_with_free_memory(self):
        self.devices()
        runtime = Hardware({})
        self.assertEqual(runtime.device, 'cuda:1')
        self.assertEqual(runtime.backend, 'CUDA')
        self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
        self.assertFalse(torch.backends.cudnn.allow_tf32)

    def test_explicit_device_is_honored(self):
        self.devices()
        self.assertEqual(Hardware({'PLATEAU_DEVICE':'cuda:0'}).device, 'cuda:0')
        self.assertEqual(Hardware({'PLATEAU_DEVICE':'cpu'}).device, 'cpu')

    def test_failed_gpu_is_skipped(self):
        self.devices(failed=('cuda:1',))
        self.assertEqual(Hardware({}).device, 'cuda:0')

    def test_mps_and_cpu_fallbacks(self):
        self.devices(free=(), mps=True)
        self.assertEqual(Hardware({}).device, 'mps')
        with patch('hardware.probe_device', side_effect=RuntimeError('kernel unavailable')):
            self.assertEqual(Hardware({}).device, 'cpu')

    def test_cpu_only_and_bad_overrides(self):
        self.devices(free=())
        self.assertEqual(Hardware({}).device, 'cpu')
        for value in ('cuda', 'mps', 'cuda:-1', 'not-a-device'):
            with self.assertRaises(ValueError): Hardware({'PLATEAU_DEVICE':value})
        for value in ('0', '-1', '65', 'abc'):
            with self.assertRaises(ValueError): Hardware({'PLATEAU_BATCH_SIZE':value})

    def test_rocm_uses_cuda_api_with_correct_label(self):
        self.devices(hip='6.4')
        self.assertEqual(Hardware({}).backend, 'ROCm')

    def test_batches_fit_memory_and_preserve_mps_policy(self):
        self.devices(free=(16 * GB,))
        config = SimpleNamespace(model_type='qwen3', hidden_size=2048, num_attention_heads=16, vocab_size=151936)
        runtime = Hardware({})
        large = runtime.batch_size(config, 256, 41)
        with patch('torch.cuda.mem_get_info', return_value=(512 * 1024 ** 2, 24 * GB)):
            small = runtime.batch_size(config, 256, 41)
        self.assertGreater(large, small)
        self.assertEqual(small, 1)
        runtime.device = 'mps'
        self.assertEqual(runtime.batch_size(config, 256, 41), 1)

    def test_only_memory_failures_qualify_for_retry(self):
        self.assertTrue(is_out_of_memory(torch.OutOfMemoryError('test')))
        self.assertTrue(is_out_of_memory(RuntimeError('MPS backend out of memory')))
        self.assertFalse(is_out_of_memory(RuntimeError('invalid tensor shape')))

    def test_model_loading_falls_back_only_in_auto_mode(self):
        from engine import Engine
        for requested in ('auto', 'mps'):
            runtime = Hardware({'PLATEAU_DEVICE':'cpu'})
            runtime.device, runtime.requested = 'mps', requested
            runtime.clear_cache = MagicMock()
            model = MagicMock()
            model.config = SimpleNamespace()
            model.to.side_effect = torch.OutOfMemoryError('simulated transfer failure')
            with patch('engine.Hardware', return_value=runtime), \
                 patch('engine.cached_snapshot', return_value=Path('/tmp/test-model-revision')), \
                 patch('engine.AutoTokenizer.from_pretrained'), \
                 patch('engine.AutoModelForCausalLM.from_pretrained', return_value=model):
                engine = Engine()
                if requested == 'auto':
                    engine.load('pythia-160m', lambda *_: None)
                    self.assertEqual(engine.device, 'cpu')
                    model.cpu.assert_called_once()
                else:
                    with self.assertRaises(torch.OutOfMemoryError):
                        engine.load('pythia-160m', lambda *_: None)
                    model.cpu.assert_not_called()


if __name__ == '__main__':
    unittest.main()
