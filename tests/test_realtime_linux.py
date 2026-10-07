import unittest
from unittest.mock import patch
from transcriptex.realtime_linux import capture_command, run_linux
from transcriptex.realtime import RealtimeConfig

class LinuxRealtimeTests(unittest.TestCase):
    def test_meeting_stream_is_isolated_without_shell(self):
        command = capture_command('headset.monitor', 737)
        self.assertIn('--monitor-stream=737', command)
        self.assertIn('--device=headset.monitor', command)
        self.assertIn('--rate=16000', command)
        self.assertNotIn('--monitor-stream=-1', capture_command('microphone'))

    def test_listing_never_initializes_whisper(self):
        config = RealtimeConfig('small', 'cpu', 'int8', 'es', 1, list_devices=True)
        with patch('transcriptex.realtime_linux.subprocess.run') as run, patch('transcriptex.realtime.RealtimeTranscriber') as model:
            self.assertTrue(run_linux(config)['success'])
            model.assert_not_called()
            run.assert_called_once_with(['pactl', 'list', 'short', 'sources'], check=True)

    def test_requires_output_and_rejects_windows_process_option(self):
        config = RealtimeConfig('small', 'cpu', 'int8', 'es', 1)
        with self.assertRaises(ValueError):
            run_linux(config)
        config.process = 'firefox'
        with self.assertRaises(ValueError):
            run_linux(config)
