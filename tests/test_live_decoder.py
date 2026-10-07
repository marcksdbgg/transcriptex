import unittest
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np

from transcriptex.live_decoder import LiveDecoder, SAMPLE_RATE
from transcriptex.realtime import RealtimeConfig
from transcriptex.realtime_linux import validate_config


class LiveDecoderTests(unittest.TestCase):
    def config(self, **kwargs):
        config = RealtimeConfig('large-v3-turbo','cpu','int8','es',1,output='live.txt',preview=False)
        for key, value in kwargs.items():
            setattr(config, key, value)
        return config

    def model(self, words):
        model = Mock()
        def decode(*args, **kwargs):
            return iter([SimpleNamespace(words=words)] if words else []), SimpleNamespace()
        model.transcribe.side_effect = decode
        return model

    def test_preview_is_not_committed_and_pause_confirms(self):
        word = SimpleNamespace(start=0.1, end=1, word=' Hola')
        decoder = LiveDecoder(self.model([word]), self.config(preview=True, chunk_duration=1))
        preview = decoder.feed(np.ones(SAMPLE_RATE, dtype=np.float32)*0.1)
        self.assertFalse(preview[0]['final'])
        self.assertEqual(decoder.committed_until, 0)
        confirmed = decoder.feed(np.zeros(int(SAMPLE_RATE*0.6), dtype=np.float32))
        self.assertTrue(confirmed[0]['final'])
        self.assertEqual(confirmed[0]['text'], 'Hola')
        self.assertEqual(decoder.flush(), [])

    def test_forced_cut_holds_boundary_word_and_flush_writes_it_once(self):
        word1=SimpleNamespace(start=0.1,end=2,word=' Hola')
        word2=SimpleNamespace(start=4.2,end=4.8,word=' mundo')
        decoder=LiveDecoder(self.model([word1,word2]),self.config(max_segment_duration=5,chunk_duration=1))
        confirmed=decoder.feed(np.ones(5*SAMPLE_RATE,dtype=np.float32)*0.1)
        self.assertEqual(confirmed[0]['text'],'Hola')
        # Subsequent window includes old audio plus the uncommitted boundary word.
        decoder.model=self.model([SimpleNamespace(start=0,end=1,word=' Hola'),SimpleNamespace(start=3.2,end=3.8,word=' mundo')])
        tail=decoder.flush()
        self.assertEqual(tail[0]['text'],'mundo')

    def test_silence_and_unrecognized_noise_buffers_are_bounded(self):
        model=self.model([])
        decoder=LiveDecoder(model,self.config(max_segment_duration=5,chunk_duration=1))
        for _ in range(300):
            decoder.feed(np.zeros(int(.2*SAMPLE_RATE),dtype=np.float32))
        model.transcribe.assert_not_called()
        self.assertLessEqual(len(decoder.buffer),int(.4*SAMPLE_RATE))
        for _ in range(200):
            decoder.feed(np.ones(int(.2*SAMPLE_RATE),dtype=np.float32)*.1)
            self.assertLessEqual(len(decoder.buffer),5*SAMPLE_RATE)

    def test_flush_preserves_partial_utterance_at_stop(self):
        decoder=LiveDecoder(self.model([SimpleNamespace(start=0,end=.4,word=' final')]),self.config())
        self.assertEqual(decoder.feed(np.ones(int(.5*SAMPLE_RATE),dtype=np.float32)*.1),[])
        self.assertEqual(decoder.flush()[0]['text'],'final')
        self.assertEqual(decoder.flush(),[])

    def test_invalid_settings_fail_before_capture(self):
        for options in ({'chunk_duration':float('nan')},{'queue_seconds':0},{'max_segment_duration':31},{'overlap_duration':10},{'microphone':'@DEFAULT_MONITOR@'}):
            with self.subTest(options=options),self.assertRaises(ValueError):
                validate_config(self.config(**options))


if __name__ == '__main__':
    unittest.main()
