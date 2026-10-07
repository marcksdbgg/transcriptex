import contextlib
import io
import json
import tempfile
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, Mock

from transcriptex.cli import main, _emit_report
from transcriptex.transcribe_single import SingleTranscriber, TranscribeSingleConfig
from transcriptex.subtitle_segments import parse_subtitle_segments


class AutomationTests(unittest.TestCase):
    def test_help_and_live_config_import_without_site_packages(self):
        repo=Path(__file__).resolve().parents[1]
        command=[sys.executable,'-S','-m','transcriptex','--help']
        result=subprocess.run(command,cwd=repo,capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stderr)
        result=subprocess.run([sys.executable,'-S','-c','from transcriptex.realtime import RealtimeConfig'],cwd=repo,capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_json_stdout_is_one_document_despite_nested_diagnostics(self):
        def handler(args):
            print('Nested model/download progress')
            return _emit_report(args, 'transcribe', {'success': True}, None, 0)
        out, err = io.StringIO(), io.StringIO()
        with patch.dict('transcriptex.cli._DISPATCH', {'transcribe': handler}), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            self.assertEqual(main(['transcribe', 'file.wav', '--json']), 0)
        self.assertTrue(json.loads(out.getvalue())['success'])
        self.assertIn('Nested model', err.getvalue())
        self.assertNotIn('Nested model', out.getvalue())

    def test_json_error_before_handler_work_is_parseable(self):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            self.assertEqual(main(['transcribe', 'x.wav', '--batch-size', '-1', '--json']), 1)
        self.assertEqual(json.loads(out.getvalue())['error_type'], 'ValueError')

    def test_json_interrupt_preserves_exit_code(self):
        out = io.StringIO()
        with patch.dict('transcriptex.cli._DISPATCH', {'transcribe': Mock(side_effect=KeyboardInterrupt)}), contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(['transcribe', 'x.wav', '--json']), 130)
        self.assertEqual(json.loads(out.getvalue())['exit_code'], 130)

    def test_multi_file_reuses_engine_and_does_not_overwrite_same_stem(self):
        calls = []
        def transcribe(config, engine):
            calls.append((config, engine))
            return {'success': True}
        with tempfile.TemporaryDirectory() as dest, patch('transcriptex.transcribe_single.run_transcribe_single', side_effect=transcribe), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(['transcribe', '/one/audio.wav', '/two/audio.wav', '-o', dest, '--json']), 0)
        self.assertIs(calls[0][1], calls[1][1])
        self.assertNotEqual(calls[0][0].output_stem, calls[1][0].output_stem)

    def test_multi_file_partial_failure_reports_warning(self):
        out = io.StringIO()
        with patch('transcriptex.transcribe_single.run_transcribe_single', side_effect=[{'success': True}, RuntimeError('bad input')]), contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(['transcribe', 'a.wav', 'b.wav', '--json']), 2)
        self.assertEqual(json.loads(out.getvalue())['report']['failed'], 1)


class SingleTranscriberTests(unittest.TestCase):
    def config(self, root, **kwargs):
        config = TranscribeSingleConfig('https://youtube.com/watch?v=jNQXAC9IVRw', Path(root), 'json',
            'large-v3-turbo', 'auto', 'auto', 'es', 5, False, quiet=True)
        for name, value in kwargs.items():
            setattr(config, name, value)
        return config

    def test_subtitles_avoid_download_and_model(self):
        payload = b'{"events":[{"tStartMs":0,"dDurationMs":1000,"segs":[{"utf8":"Hola mundo"}]}]}'
        with tempfile.TemporaryDirectory() as root:
            engine = Mock()
            t = SingleTranscriber(self.config(root), engine)
            t.client = Mock()
            t.client.extract_video_info.return_value = {'duration':1,'subtitles':{'es':[{'url':'track','ext':'json3'}]}}
            t.client.fetch_binary.return_value = payload
            report = t.run()
            self.assertEqual(report['source'], 'manual')
            self.assertEqual(json.loads(Path(report['output_file']).read_text())[0]['text'], 'Hola mundo')
            engine.transcribe_file.assert_not_called()
            t.client.download_audio.assert_not_called()

    def test_subtitles_only_fails_without_silently_using_asr(self):
        with tempfile.TemporaryDirectory() as root:
            engine = Mock()
            t = SingleTranscriber(self.config(root, source='subtitles'), engine)
            t.client.extract_video_info = Mock(return_value={})
            with self.assertRaisesRegex(ValueError, 'No subtitle'):
                t.run()
            engine.transcribe_file.assert_not_called()

    def test_auto_subtitle_failure_falls_back_to_asr(self):
        with tempfile.TemporaryDirectory() as root:
            engine = Mock(device='cpu')
            engine.transcribe_file.return_value = ([SimpleNamespace(start=0,end=1,text=' recognized')], SimpleNamespace(language='es',duration=1),1)
            t = SingleTranscriber(self.config(root), engine)
            t.client = Mock()
            t.client.extract_video_info.return_value = {}
            t.client.download_audio.return_value = Path('download.webm')
            report = t.run()
            self.assertEqual(report['source'], 'asr')
            self.assertIn('No subtitle', report['subtitle_fallback_reason'])
            engine.transcribe_file.assert_called_once()

    def test_failed_export_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as root:
            t = SingleTranscriber(self.config(root))
            t._try_subtitles = Mock(return_value=([SimpleNamespace(start=0,end=1,text='new')], 'es', 'manual', 1))
            output = Path(root)/'transcript_jNQXAC9IVRw.json'
            output.write_text('original')
            def fail(segments, path):
                path.write_text('partial')
                raise OSError('disk full')
            t._write_json = fail
            with self.assertRaises(OSError):
                t.run()
            self.assertEqual(output.read_text(), 'original')
            self.assertEqual(list(Path(root).glob('*.tmp')), [])

    def test_timestamp_rounding_carries_to_minutes_and_hours(self):
        t = SingleTranscriber(self.config('.'))
        self.assertEqual(t._format_timestamp(59.9996), '00:01:00,000')
        self.assertEqual(t._format_timestamp(3599.9996, '.'), '01:00:00.000')

    def test_timed_parsers_preserve_cues(self):
        vtt = b'WEBVTT\n\n00:00:01.000 --> 00:00:03.000 align:start\nHola <b>mundo</b>\n\n'
        srt = b'1\n00:00:01,000 --> 00:00:03,000\nHola mundo\n\n'
        for data, ext in ((vtt, 'vtt'), (srt, 'srt'), (b'<transcript><text start="1" dur="2">Hola mundo</text></transcript>', 'xml')):
            segments = parse_subtitle_segments(data, ext)
            self.assertEqual((segments[0].start, segments[0].end, segments[0].text), (1,3,'Hola mundo'))
        with self.assertRaises(ValueError):
            parse_subtitle_segments(b'WEBVTT\n\n', 'vtt')


if __name__ == '__main__':
    unittest.main()
