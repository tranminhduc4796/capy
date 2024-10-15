from collections import deque
from threading import Thread, Event
from queue import Queue
import audioop

import pyaudio as pa
import numpy as np

from faster_whisper import WhisperModel


class CapyASR:
    """
    Capy's Automatic Speech Recognition (ASR) engine, transcribe audio from microphone in real-time.

    The implementation is inspired by the speech_recognition library.
    https://github.com/Uberi/speech_recognition

    Features:
    - Record audio from microphone in real-time
    - Transcribe audio from microphone in real-time
    - TODO: Detect hot word to start the recording
    - Prevent silence at the begining, avoid redundunt transcribing computations
    - Auto stop recording if the silence is detected after a certain amount of time
    """

    def __init__(self, sample_rate=16000,
                 frames_per_buffer=1024,
                 energy_threshold=300,
                 end_silence_duration=0.8,
                 non_speak_duration=0.5
                 ):
        """
        Initialize the ASR engine with the given parameters.

        Args:
            sample_rate (int): Sample rate of the audio, the higher the better, but more computationally expensive.
            frames_per_buffer (int): The number of frames to read at once.
            energy_threshold (int): The minimum energy of the audio signal to be considered as a speech.
            end_silence_duration (float): The duration of silence to detect the end of a phrase.
            non_speak_duration (float): The duration of non-speech to include on both sides of the phrase.
        """
        self.audio = pa.PyAudio()
        self.audio_format = pa.paInt16  # For effeciency
        try:
            # Check the stream then close it
            self.stream = self.audio.open(
                format=self.audio_format,
                channels=1,
                rate=sample_rate,
                input=True,
                frames_per_buffer=frames_per_buffer,
            )
            self.stream.stop_stream()
        except Exception as e:
            self.audio.terminate()
            raise e

        assert non_speak_duration < end_silence_duration, "non_speak_duration must be less than end_silence_duration"

        self.energy_threshold = energy_threshold
        self.end_silence_duration = end_silence_duration
        self.frames_per_buffer = frames_per_buffer
        self.sample_rate = sample_rate
        self.non_speak_duration = non_speak_duration

        # For recording
        self.finish_record_event = Event()
        self.audio_queue = Queue()
        self.listener_thread = None

        # For transcribing
        self.device = "cpu"
        self.compute_type = "int8"
        self.model = WhisperModel(
            model_size_or_path="tiny.en",
            device=self.device,
            compute_type=self.compute_type,
            num_workers=12,
            cpu_threads=8,
            local_files_only=True,
        )

    def __del__(self):
        self.audio.terminate()

    def calibrate_energy_threshold(self, dynamic_energy_adjustment_damping=0.15, dynamic_energy_ratio=1.1, duration=1):
        """
        Adjusts the energy threshold dynamically with the ambient energy level using asymmetric weighted average.
        NOTE: This method should be called only on periods of audio without speech for accurate result. 
        """
        seconds_per_buffer = self.frames_per_buffer / self.sample_rate
        elapsed_time = 0
        
        if self.stream.is_stopped():
            self.stream.start_stream()

        while True:
            elapsed_time += seconds_per_buffer
            if elapsed_time > duration:
                break
            buffer = self.stream.read(self.frames_per_buffer)
            energy = audioop.rms(buffer, pa.get_sample_size(self.audio_format))

            # dynamically adjust the energy threshold using asymmetric weighted average
            # account for different chunk sizes and rates
            damping = dynamic_energy_adjustment_damping ** seconds_per_buffer
            target_energy = energy * dynamic_energy_ratio
            self.energy_threshold = self.energy_threshold * \
                damping + target_energy * (1 - damping)

        return self.energy_threshold

    def listen(self, timeout=5):
        """
        Record and detect a phrase from the microphone.
        A phrase is a continuous segment of audio.
        It starts when the audio's energy is above the `energy_threshold` 
        and end when `timeout` seconds have passed or 
        the detected silence lasts for `end_silence_duration` seconds.
        A phrase can be also bounded by silence segments if the silence segments exists.

        Args:
            timeout (int, optional): The maximum number of seconds for a phrase. Defaults to 5.
        Returns:
            bytes: The recorded audio in bytes, in int16 format.
        """
        seconds_per_buffer = self.frames_per_buffer / self.sample_rate
        # The number of non-speak frames bound around a phrase
        n_non_speak_frames = int(self.non_speak_duration / seconds_per_buffer)

        elapsed_time = 0
        buffer = b""  # End of the recording is when buffer empty
        frames = deque()

        if self.stream.is_stopped():
            self.stream.start_stream()
        
        # Wait until detecting the start of the phrase
        while True:
            buffer = self.stream.read(self.frames_per_buffer)
            if len(buffer) == 0:
                break

            # Raise error if waiting too long
            elapsed_time += seconds_per_buffer
            if elapsed_time > timeout:
                raise TimeoutError(
                    f"No speech detected in {timeout} seconds.")

            # Only keep `non_speak_duration` seconds of non-speak audio at the begining of a phrase
            frames.append(buffer)
            if len(frames) > n_non_speak_frames:
                frames.popleft()

            energy = audioop.rms(
                buffer, pa.get_sample_size(self.audio_format))
            if energy > self.energy_threshold:
                # Start of phrase detected
                break

        # Recording
        pause_secs = 0  # Number of seconds of silence detected after the start
        n_end_silence_frames = 0  # Number of silence frames detected by the end of the phrase
        start_time = elapsed_time
        while True:
            elapsed_time += seconds_per_buffer

            if elapsed_time - start_time >= timeout:
                break

            buffer = self.stream.read(self.frames_per_buffer)
            if len(buffer) == 0:
                break
            frames.append(buffer)

            energy = audioop.rms(
                buffer, pa.get_sample_size(self.audio_format))
            if energy < self.energy_threshold:
                # Silence detected
                pause_secs += seconds_per_buffer
                n_end_silence_frames += 1
            else:
                pause_secs = n_end_silence_frames = 0
            if pause_secs >= self.end_silence_duration:
                # End of phrase detected
                break

        # Remove last silence frames
        for _ in range(n_end_silence_frames - n_non_speak_frames):
            frames.pop()

        return b"".join(frames)

    def listen_in_background(self, timeout=5):
        """
        Record and detect a phrase from the microphone in a background thread.

        Args:
            timeout (int, optional): The maximum number of seconds for a phrase. Defaults to 5.
        """

        def threaded_listen():
            while not self.finish_record_event.is_set():
                try:
                    audio = self.listen(timeout=timeout)
                except TimeoutError:
                    pass
                else:
                    if not self.finish_record_event.is_set():
                        self.audio_queue.put(audio)

        self.listener_thread = Thread(target=threaded_listen)
        self.listener_thread.daemon = True
        self.listener_thread.start()

    def _transcribe(self, audio):
        """
        Transcribe the audio in the given np.ndarray of float32.

        Args:
            audio (bytes): The audio array.
        Returns:
            str: The transcribed text.
        """
        segments, _ = self.model.transcribe(
                audio,
                beam_size=3,
                language="en",
                vad_filter=True,
                without_timestamps=True,
                condition_on_previous_text=True,
            )
        return "".join([segment.text for segment in segments])

    def transcribe(self):
        """
        Transcribe the audio from microphone. Transcript chunks will be yielded.
        """
        if not self.audio_queue.empty():
            audio = b"".join(self.audio_queue.queue)
            self.audio_queue.queue.clear()
            # Convert 16-bit int to 32-bit float and clamp to [-1.0, 1.0]
            float32_audio = np.frombuffer(
                audio, dtype=np.int16).astype(np.float32) / 32768.0
            transcript = self._transcribe(float32_audio)
            return transcript
        # Empty string when no audio yet
        return ""
    
    def stop_transcribing(self):
        self.finish_record_event.set()
        self.listener_thread.join()
        self.listener_thread = None
        self.finish_record_event.clear()
        self.stream.stop_stream()
