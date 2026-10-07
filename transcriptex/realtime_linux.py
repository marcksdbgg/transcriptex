"""PipeWire/PulseAudio capture with bounded buffers and a final full-audio pass."""
import datetime
import json
import math
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path


def capture_command(source, stream=-1):
    command = ['parec', '--raw', '--format=float32le', '--rate=16000',
               '--channels=1', '--latency-msec=100', '--device=' + source,
               '--property=node.dont-fallback=true', '--property=node.dont-reconnect=true']
    if stream >= 0:
        command.append('--monitor-stream=' + str(stream))
    return command


def pulse_list(kind):
    result = subprocess.run(['pactl', '-f', 'json', 'list', kind], check=True,
        capture_output=True, text=True, timeout=5)
    return json.loads(result.stdout)


def verify_capture_route(pid, source_index, monitor_stream=-1):
    stream = next((s for s in pulse_list('source-outputs')
        if str(s.get('properties', {}).get('application.process.id')) == str(pid)), None)
    if stream is None:
        raise RuntimeError('Audio capture stream disappeared; stopping capture')
    if monitor_stream >= 0 and stream.get('driver') == 'PipeWire':
        # PipeWire represents a sink-input monitor as source PA_INVALID_INDEX.
        # Its actual target is the selected playback node, not a source node.
        props = stream.get('properties', {})
        matches = str(props.get('target.object')) == str(monitor_stream) and str(props.get('stream.capture.sink')).lower() == 'true'
    else:
        matches = str(stream.get('source')) == str(source_index)
    if not matches:
        raise RuntimeError('Audio capture route differs from the selected source; stopping capture')


def resolve_sources(config):
    """Resolve aliases and reject nonexistent targets before starting capture.

    PipeWire may otherwise route an unknown PulseAudio name to the default source.
    """
    sources = pulse_list('sources')
    def resolve(name):
        if name == '@DEFAULT_MONITOR@':
            default = subprocess.check_output(['pactl', 'get-default-sink'], text=True, timeout=5).strip()
            sink = next((s for s in pulse_list('sinks') if s['name'] == default), None)
            if sink is None:
                raise ValueError('Default sink unavailable')
            name = sink.get('monitor_source_name') or sink.get('monitor_source')
            if isinstance(name, int):
                name = str(name)
        elif name == '@DEFAULT_SOURCE@':
            name = subprocess.check_output(['pactl', 'get-default-source'], text=True, timeout=5).strip()
        source = next((s for s in sources if s['name'] == name or str(s['index']) == name), None)
        if source is None:
            raise ValueError('Audio source not found: ' + name + '. Use --list-devices.')
        return source

    system = resolve(config.source)
    microphone = resolve(config.microphone) if config.microphone else None
    if microphone is not None and microphone['index'] == system['index']:
        raise ValueError('Microphone and system source resolve to the same input')
    if config.monitor_stream >= 0:
        stream = next((s for s in pulse_list('sink-inputs') if s['index'] == config.monitor_stream), None)
        if stream is None:
            raise ValueError('Meeting stream not found; select its current sink-input index')
        sink = next((s for s in pulse_list('sinks') if s['index'] == stream['sink']), None)
        monitor = (sink.get('monitor_source_name') or sink.get('monitor_source')) if sink else None
        if monitor not in (system['name'], system['index'], str(system['index'])):
            raise ValueError('Selected source is not the monitor of the meeting stream sink')
    return system['name'], microphone['name'] if microphone else ''


def validate_config(config):
    if config.process:
        raise ValueError('Linux: use --monitor-stream with the Meet sink-input index, not --process.')
    if not config.output:
        raise ValueError('Linux realtime requires --output transcript.txt')
    for name in ('chunk_duration', 'max_segment_duration', 'silence_duration',
                 'silence_threshold', 'overlap_duration', 'queue_seconds'):
        value = getattr(config, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(name + ' must be a finite positive number')
    if config.chunk_duration < 0.5:
        raise ValueError('chunk_duration must be at least 0.5 s')
    if config.max_segment_duration < max(3.0, config.chunk_duration + 1):
        raise ValueError('max_segment_duration must exceed the preview interval by at least 1 s')
    if config.max_segment_duration > 30:
        raise ValueError('max_segment_duration cannot exceed the Whisper context window (30 s)')
    if config.overlap_duration * 2 >= config.max_segment_duration:
        raise ValueError('overlap_duration must be less than half max_segment_duration')
    if config.queue_seconds < 1 or config.queue_seconds > 300:
        raise ValueError('queue_seconds must be between 1 and 300')
    if config.microphone and config.microphone == config.source:
        raise ValueError('Microphone and system source must differ (otherwise audio is duplicated)')


def run_linux(config):
    if config.list_devices:
        subprocess.run(['pactl', 'list', 'short', 'sources'], check=True)
        return {'success': True}
    validate_config(config)
    if shutil.which('parec') is None:
        raise RuntimeError('parec is required; install PulseAudio client utilities for PipeWire/PulseAudio.')
    source_name, microphone_name = resolve_sources(config)
    from transcriptex.asr_engine import ASREngine
    engine = ASREngine(config.model, config.device, config.compute_type)
    model = engine.load()
    import numpy as np
    from transcriptex.live_decoder import LiveDecoder, SAMPLE_RATE
    sources = [('Meet' if config.monitor_stream >= 0 else 'Sistema', source_name, config.monitor_stream)]
    if microphone_name:
        sources.append(('Micrófono local', microphone_name, -1))
    output = Path(config.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    frame_s = 0.2
    chunks = queue.Queue(maxsize=max(5, int(config.queue_seconds / frame_s) * len(sources)))
    errors = queue.SimpleQueue()
    processes, readers = [], []
    process_lock = threading.Lock()
    started = time.monotonic()
    wall_started = datetime.datetime.now().astimezone()
    decoders = {label: LiveDecoder(model, config) for label, _, _ in sources}
    captured_samples = {label: 0 for label, _, _ in sources}
    written = 0
    backlog_warnings = 0
    last_warning = 0.0
    capture_error = None
    final_file = None
    last_received = {label: started for label, _, _ in sources}

    def reader(label, source, stream):
        attempts = 0
        while not stop.is_set():
            process = None
            try:
                selected = next((s for s in pulse_list('sources') if s['name'] == source), None)
                if selected is None:
                    raise RuntimeError('Audio source disappeared: ' + source)
                process = subprocess.Popen(capture_command(source, stream), stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, start_new_session=True)
                with process_lock:
                    processes.append(process)
                source_start = time.monotonic() - started
                source_samples = 0
                last_route_check = 0.0
                while not stop.is_set():
                    data = process.stdout.read(int(frame_s * SAMPLE_RATE) * 4)
                    if not data:
                        break
                    if len(data) % 4:
                        raise RuntimeError('Incomplete PCM sample from ' + source)
                    received_at = time.monotonic()
                    if received_at - last_route_check >= 1:
                        verify_capture_route(process.pid, selected['index'], stream)
                        last_route_check = received_at
                    last_received[label] = received_at
                    stamp = source_start + source_samples / SAMPLE_RATE
                    source_samples += len(data) // 4
                    try:
                        chunks.put((label, data, stamp), timeout=1)
                    except queue.Full:
                        raise RuntimeError('Audio queue overflow; reduce preview frequency or use a smaller model. Final audio may be incomplete.')
                if stop.is_set():
                    break
                process.wait(timeout=3)
                detail = process.stderr.read(4096).decode('utf-8', errors='replace').strip()
                if stream >= 0:
                    raise RuntimeError('Isolated meeting stream ended. Select its new sink-input index and restart; capture will not switch to all system audio. ' + detail)
                attempts = attempts + 1 if source_samples < 5 * SAMPLE_RATE else 1
                if attempts > 3:
                    raise RuntimeError('Audio source could not be recovered: ' + source + '. ' + detail)
                print('[WARN] Audio source ended; retrying ' + source, file=sys.stderr, flush=True)
                stop.wait(min(2.0, 0.5 * attempts))
            except Exception as exc:
                if not stop.is_set():
                    errors.put(exc)
                break
            finally:
                if process is not None:
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                    process.stdout.close()
                    process.stderr.close()
                    with process_lock:
                        processes.remove(process)

    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with tempfile.TemporaryDirectory(prefix='transcriptex-live-') as temporary, os.fdopen(fd, 'a', encoding='utf-8', buffering=1) as transcript:
        recordings = {}
        if config.finalize or config.record_audio:
            for index, (label, _, _) in enumerate(sources):
                if config.record_audio:
                    fd_audio, name = tempfile.mkstemp(prefix=output.stem + '-source' + str(index) + '-', suffix='.wav', dir=output.parent)
                    os.close(fd_audio)
                    path = Path(name)
                else:
                    path = Path(temporary) / (str(index) + '.wav')
                writer = wave.open(str(path), 'wb')
                writer.setnchannels(1)
                writer.setsampwidth(2)
                writer.setframerate(SAMPLE_RATE)
                recordings[label] = (path, writer)

        def write(text):
            transcript.write(text + '\n')
            transcript.flush()
            os.fsync(transcript.fileno())
            print(text, flush=True)

        def emit(label, events):
            nonlocal written
            for event in events:
                stamp = (wall_started + datetime.timedelta(seconds=event['start'])).isoformat(timespec='seconds')
                if event['final']:
                    write('[' + stamp + '] [' + label + '] ' + event['text'])
                    written += 1
                else:
                    print('[Parcial] [' + label + '] ' + event['text'], flush=True)

        def consume(item):
            label, data, stamp = item
            samples = np.frombuffer(data, dtype='<f4')
            if not np.isfinite(samples).all():
                raise RuntimeError('Non-finite PCM samples from ' + label)
            decoder = decoders[label]
            expected = captured_samples[label] / SAMPLE_RATE
            # Preserve missing capture time on reconnect; never concatenate across a gap.
            gap = max(0, round((stamp - expected) * SAMPLE_RATE))
            if gap > int(0.1 * SAMPLE_RATE):
                emit(label, decoder.flush())
                decoder.total_samples += gap
                decoder.offset += gap / SAMPLE_RATE
                if label in recordings:
                    writer = recordings[label][1]
                    remaining = gap
                    while remaining:
                        count = min(remaining, SAMPLE_RATE)
                        writer.writeframesraw(b'\x00\x00' * count)
                        remaining -= count
                captured_samples[label] += gap
            if label in recordings:
                recordings[label][1].writeframesraw((np.clip(samples, -1, 1) * 32767).astype('<i2').tobytes())
            captured_samples[label] += len(samples)
            emit(label, decoder.feed(samples))

        write('\n# Transcripción automática iniciada ' + wall_started.isoformat(timespec='seconds'))
        write('# Fuentes: ' + ', '.join(label for label, _, _ in sources) + '. Etiquetas de fuente, no identificación de hablantes.')
        write('# Texto [Parcial] solo en consola; TXT contiene segmentos confirmados. Versión final: ' + str(config.finalize))
        try:
            for label, source, stream in sources:
                thread = threading.Thread(target=reader, args=(label, source, stream), daemon=True)
                readers.append(thread)
                thread.start()
            print('[OK] Capturando. TXT: ' + str(output) + ' | Ctrl+C para detener.', flush=True)
            while True:
                if not errors.empty():
                    raise errors.get()
                stalled = [label for label, received in last_received.items() if time.monotonic() - received > 5]
                if stalled:
                    raise RuntimeError('No PCM audio received for 5 s from: ' + ', '.join(stalled))
                try:
                    item = chunks.get(timeout=0.2)
                except queue.Empty:
                    continue
                consume(item)
                if chunks.qsize() * frame_s / len(sources) > 6 and time.monotonic() - last_warning > 5:
                    last_warning = time.monotonic()
                    backlog_warnings += 1
                    print('[WARN] Transcripción retrasada; bloques pendientes: ' + str(chunks.qsize()), file=sys.stderr, flush=True)
        except KeyboardInterrupt:
            print('[INFO] Deteniendo captura y confirmando audio pendiente...', file=sys.stderr, flush=True)
        except Exception as exc:
            capture_error = str(exc)
            write('# ERROR de captura: ' + capture_error)
        finally:
            stop.set()
            with process_lock:
                for process in processes:
                    if process.poll() is None:
                        process.terminate()
            for thread in readers:
                thread.join(timeout=5)
            try:
                while not chunks.empty():
                    consume(chunks.get_nowait())
                for label, decoder in decoders.items():
                    emit(label, decoder.flush())
            finally:
                for _, writer in recordings.values():
                    writer.close()

        if config.finalize and recordings:
            print('[INFO] Generando transcripción final con el audio completo...', file=sys.stderr, flush=True)
            events = []
            try:
                for label, (path, _) in recordings.items():
                    if captured_samples[label] == 0:
                        continue
                    segments, _, _ = engine.transcribe_file(str(path), language=config.language or None, beam_size=config.beam_size)
                    for segment in segments:
                        if segment.text.strip():
                            events.append((segment.start, label, segment.text.strip()))
                events.sort(key=lambda event: event[0])
                final_file = output.with_suffix('.final.txt')
                if final_file == output:
                    final_file = output.with_name(output.stem + '.complete.txt')
                fd_final, name = tempfile.mkstemp(prefix='.' + output.stem + '-', suffix='.tmp', dir=output.parent)
                try:
                    with os.fdopen(fd_final, 'w', encoding='utf-8') as final:
                        final.write('# Transcripción final; etiquetas de fuente, no identificación de hablantes.\n')
                        if capture_error:
                            final.write('# Grabación incompleta por error: ' + capture_error + '\n')
                        for start, label, text in events:
                            stamp = (wall_started + datetime.timedelta(seconds=start)).isoformat(timespec='seconds')
                            final.write('[' + stamp + '] [' + label + '] ' + text + '\n')
                    os.replace(name, final_file)
                finally:
                    Path(name).unlink(missing_ok=True)
                write('# Versión final guardada: ' + str(final_file))
            except Exception as exc:
                capture_error = capture_error or ('Final transcription failed: ' + str(exc))
                write('# ERROR de versión final: ' + str(exc))
                final_file = None
        write('# Captura detenida. El último frame de audio puede quedar incompleto.')
        return {'success': capture_error is None, 'output': str(output), 'final_output': str(final_file) if final_file else None,
            'error': capture_error, 'confirmed_segments': written, 'backlog_warnings': backlog_warnings,
            'recordings': [str(path) for path, _ in recordings.values()] if config.record_audio else []}
