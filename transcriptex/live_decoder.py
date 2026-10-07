"""Pause-aware, bounded streaming recognition shared by live capture and replay.

Previews are provisional. Only pause-ended or time-limited segments are committed.
Forced cuts retain audio and use word timings to avoid writing the overlap twice.
"""
import numpy as np

SAMPLE_RATE = 16000


class LiveDecoder:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.buffer = np.empty(0, dtype=np.float32)
        self.offset = 0.0
        self.total_samples = 0
        self.quiet_seconds = 0.0
        self.has_speech = False
        self.last_preview_at = 0.0
        self.committed_until = 0.0
        self.history = ''
        self.last_preview = ''

    def _decode(self, final=False, forced=False):
        if not self.has_speech or not len(self.buffer):
            return None
        segments, _ = self.model.transcribe(self.buffer, language=self.config.language or None,
            beam_size=self.config.beam_size, vad_filter=True,
            vad_parameters={'min_silence_duration_ms': 500, 'speech_pad_ms': 200},
            condition_on_previous_text=False, initial_prompt=self.history[-300:] or None,
            word_timestamps=True)
        segments = list(segments)  # Decoding is lazy: consume before timing or trimming.
        end = self.offset + len(self.buffer) / SAMPLE_RATE
        cutoff = end - self.config.overlap_duration if forced else end
        parts = []
        first = None
        last = self.committed_until
        for segment in segments:
            if getattr(segment, 'words', None):
                for word in segment.words:
                    start, finish = self.offset + word.start, self.offset + word.end
                    # Half-word overlap threshold tolerates alignment drift at the boundary.
                    middle = (start + finish) / 2
                    if middle <= self.committed_until + 0.02 or finish > cutoff + 0.02:
                        continue
                    parts.append(word.word)
                    first = start if first is None else first
                    last = max(last, finish)
            else:
                start, finish = self.offset + segment.start, self.offset + segment.end
                if finish <= self.committed_until + 0.02 or finish > cutoff + 0.02:
                    continue
                parts.append(segment.text)
                first = start if first is None else first
                last = max(last, finish)
        text = ''.join(parts).strip()
        if not text:
            return None
        if final:
            self.committed_until = last
            self.history = (self.history + ' ' + text)[-600:]
            self.last_preview = ''
        elif text == self.last_preview:
            return None
        else:
            self.last_preview = text
        return {'text': text, 'start': max(0.0, first), 'end': last, 'final': final}

    def feed(self, samples):
        """Accept capture frames (typically 0.2 s), return confirmed/preview events."""
        samples = np.asarray(samples, dtype=np.float32)
        if not len(samples):
            return []
        duration = len(samples) / SAMPLE_RATE
        self.total_samples += len(samples)
        voiced = float(np.sqrt(np.mean(samples * samples))) >= self.config.silence_threshold
        self.buffer = np.concatenate((self.buffer, samples))
        if voiced:
            self.has_speech = True
            self.quiet_seconds = 0.0
        else:
            self.quiet_seconds += duration
        elapsed = self.total_samples / SAMPLE_RATE
        events = []
        pause = self.has_speech and self.quiet_seconds + 1e-6 >= self.config.silence_duration
        forced = len(self.buffer) / SAMPLE_RATE + 1e-6 >= self.config.max_segment_duration
        if pause or forced:
            event = self._decode(final=True, forced=forced and not pause)
            if event:
                events.append(event)
            end = self.offset + len(self.buffer) / SAMPLE_RATE
            if pause:
                # Keep a small pre-roll; do not carry a silent utterance indefinitely.
                keep = min(len(self.buffer), int(0.2 * SAMPLE_RATE))
                self.has_speech = False
            else:
                # Retain both context and any uncommitted final words at a forced cut.
                keep_s = min(self.config.max_segment_duration - self.config.chunk_duration,
                    max(self.config.overlap_duration * 2, end - self.committed_until + self.config.overlap_duration))
                keep = int(keep_s * SAMPLE_RATE) if self.has_speech and event else int(self.config.overlap_duration * SAMPLE_RATE)
            self.buffer = self.buffer[-keep:].copy() if keep else np.empty(0, dtype=np.float32)
            self.offset = end - len(self.buffer) / SAMPLE_RATE
            self.last_preview_at = elapsed
        elif self.config.preview and self.has_speech and elapsed - self.last_preview_at >= self.config.chunk_duration:
            event = self._decode()
            if event:
                events.append(event)
            self.last_preview_at = elapsed
        elif not self.has_speech and len(self.buffer) > int(0.4 * SAMPLE_RATE):
            self.buffer = self.buffer[-int(0.2 * SAMPLE_RATE):].copy()
            self.offset = elapsed - len(self.buffer) / SAMPLE_RATE
        return events

    def flush(self):
        event = self._decode(final=True)
        end = self.offset + len(self.buffer) / SAMPLE_RATE
        self.buffer = np.empty(0, dtype=np.float32)
        self.offset = end
        self.has_speech = False
        self.quiet_seconds = 0.0
        return [event] if event else []
