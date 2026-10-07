import unittest
from unittest.mock import patch

from transcriptex.realtime import RealtimeConfig
from transcriptex.realtime_linux import resolve_sources, run_linux, capture_command, verify_capture_route


class SourceValidationTests(unittest.TestCase):
    sources = [{'index':1,'name':'headset.monitor'}, {'index':2,'name':'microphone'}]

    def config(self, **kwargs):
        config=RealtimeConfig('small','cpu','int8','es',1,source='headset.monitor',output='live.txt')
        for key,value in kwargs.items():
            setattr(config,key,value)
        return config

    def test_unknown_source_fails_before_model_or_capture(self):
        with patch('transcriptex.realtime_linux.pulse_list',return_value=self.sources), patch('transcriptex.realtime_linux.shutil.which',return_value='/usr/bin/parec'), patch('transcriptex.asr_engine.ASREngine') as engine, patch('transcriptex.realtime_linux.subprocess.Popen') as process:
            with self.assertRaisesRegex(ValueError,'Audio source not found'):
                run_linux(self.config(source='nonexistent'))
            engine.assert_not_called()
            process.assert_not_called()

    def test_default_monitor_resolves_without_selecting_microphone(self):
        inventories={'sources':self.sources,'sinks':[{'index':3,'name':'headset','monitor_source':'headset.monitor'}]}
        with patch('transcriptex.realtime_linux.pulse_list',side_effect=lambda k:inventories[k]), patch('transcriptex.realtime_linux.subprocess.check_output',return_value='headset\n'):
            self.assertEqual(resolve_sources(self.config(source='@DEFAULT_MONITOR@')),('headset.monitor',''))

    def test_monitor_name_alias_supported_on_pulseaudio(self):
        inventories={'sources':self.sources,'sinks':[{'index':3,'name':'headset','monitor_source_name':'headset.monitor'}]}
        with patch('transcriptex.realtime_linux.pulse_list',side_effect=lambda k:inventories[k]), patch('transcriptex.realtime_linux.subprocess.check_output',return_value='headset\n'):
            self.assertEqual(resolve_sources(self.config(source='@DEFAULT_MONITOR@')),('headset.monitor',''))

    def test_missing_stream_and_wrong_sink_are_rejected(self):
        inventories={'sources':self.sources,'sink-inputs':[],'sinks':[]}
        with patch('transcriptex.realtime_linux.pulse_list',side_effect=lambda k:inventories[k]):
            with self.assertRaisesRegex(ValueError,'stream not found'):
                resolve_sources(self.config(monitor_stream=44))
            inventories['sink-inputs']=[{'index':44,'sink':3}]
            inventories['sinks']=[{'index':3,'name':'other','monitor_source':'other.monitor'}]
            with self.assertRaisesRegex(ValueError,'not the monitor'):
                resolve_sources(self.config(monitor_stream=44))

    def test_matching_isolated_stream_remains_explicit(self):
        inventories={'sources':self.sources,'sink-inputs':[{'index':44,'sink':3}],
            'sinks':[{'index':3,'name':'headset','monitor_source':'headset.monitor'}]}
        with patch('transcriptex.realtime_linux.pulse_list',side_effect=lambda k:inventories[k]):
            self.assertEqual(resolve_sources(self.config(monitor_stream=44)),('headset.monitor',''))
        command=capture_command('headset.monitor',44)
        self.assertIn('--monitor-stream=44',command)
        self.assertIn('--property=node.dont-fallback=true',command)
        self.assertIn('--property=node.dont-reconnect=true',command)

    def test_numeric_alias_cannot_duplicate_second_source(self):
        with patch('transcriptex.realtime_linux.pulse_list',return_value=self.sources):
            with self.assertRaisesRegex(ValueError,'same input'):
                resolve_sources(self.config(microphone='1'))

    def test_routing_guard_rejects_implicit_default_capture(self):
        wrong=[{'source':2,'properties':{'application.process.id':'123'}}]
        with patch('transcriptex.realtime_linux.pulse_list',return_value=wrong):
            with self.assertRaisesRegex(RuntimeError,'differs'):
                verify_capture_route(123,1)
        wrong[0]['source']=1
        with patch('transcriptex.realtime_linux.pulse_list',return_value=wrong):
            verify_capture_route(123,1)

    def test_pipewire_isolated_stream_uses_playback_target_not_source_index(self):
        stream={'source':4294967295,'driver':'PipeWire','properties':{
            'application.process.id':'123','target.object':'44','stream.capture.sink':'true'}}
        with patch('transcriptex.realtime_linux.pulse_list',return_value=[stream]):
            verify_capture_route(123,1,44)
            with self.assertRaisesRegex(RuntimeError,'differs'):
                verify_capture_route(123,1,45)


if __name__ == '__main__':
    unittest.main()
