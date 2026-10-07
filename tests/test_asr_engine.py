import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import sys

from transcriptex.asr_engine import ASREngine


class EngineTests(unittest.TestCase):
    def engine(self):
        engine = ASREngine('large-v3-turbo', 'cuda', 'float16')
        engine.model = Mock()
        return engine

    def test_generator_oom_retries_smaller_batch_before_returning(self):
        engine = self.engine()
        engine.batched = Mock()
        info = SimpleNamespace(language='es')
        def fail():
            yield 'partial'
            raise RuntimeError('CUDA out of memory')
        engine.batched.transcribe.side_effect = [(fail(), info), (iter(['complete']), info)]
        segments, _, size = engine.transcribe_file('audio.wav', batch_size=8)
        self.assertEqual(segments, ['complete'])
        self.assertEqual(size, 4)
        self.assertEqual([c.kwargs['batch_size'] for c in engine.batched.transcribe.call_args_list], [8,4])

    def test_no_vad_and_cpu_auto_use_sequential_model(self):
        engine = self.engine()
        engine.model.transcribe.return_value = (iter([]), SimpleNamespace())
        self.assertEqual(engine.transcribe_file('audio.wav', batch_size=8, vad_filter=False)[2], 1)
        engine.device = 'cpu'
        engine.model.transcribe.return_value = (iter([]), SimpleNamespace())
        self.assertEqual(engine.transcribe_file('audio.wav')[2], 1)

    def test_unrelated_error_is_not_retried(self):
        engine = self.engine()
        engine.model.transcribe.side_effect = RuntimeError('corrupt media')
        with self.assertRaisesRegex(RuntimeError, 'corrupt media'):
            engine.transcribe_file('audio.wav', batch_size=1)
        engine.model.transcribe.assert_called_once()

    def test_model_is_loaded_once_for_multiple_inputs(self):
        model = Mock()
        model.transcribe.side_effect = [(iter([]), SimpleNamespace()), (iter([]), SimpleNamespace())]
        module = SimpleNamespace(WhisperModel=Mock(return_value=model))
        with patch.dict(sys.modules, {'faster_whisper': module}), patch('transcriptex.dependency_manager.ensure_faster_whisper'):
            engine = ASREngine('small', 'cpu', 'int8')
            engine.transcribe_file('a.wav')
            engine.transcribe_file('b.wav')
        module.WhisperModel.assert_called_once()


if __name__ == '__main__':
    unittest.main()
