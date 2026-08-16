#!/usr/bin/env python3
"""Non-blocking, once-per-event voice playback for CyberDog and simulation.

The real robot backend publishes ``protocol/msg/AudioPlayExtend`` to
``speech_play_extend``.  With ``is_online=False``, the audio board resolves a
pre-installed play ID from ``/SDCARD/sound/yaml/sound.toml`` and plays the
corresponding OPUS file.  The local backend exists only for simulation and
uses ``aplay`` without invoking a shell.

This module intentionally does not import ``protocol`` at file import time so
the same control program can still start in a simulation image that lacks the
real-robot message package.
"""

import os
import queue
import shutil
import subprocess
import threading
import time
from typing import Any, Dict, Mapping, Optional, Set, Tuple, Type


DEFAULT_EVENT_PLAY_IDS: Dict[str, int] = {
    "bar": 50001,
    "obstacle": 50002,
    "cola": 50003,
    "orange_ball": 50004,
    "football": 50005,
    # Names used by the current detector implementation.
    "blue_ball": 50004,
    "white_ball": 50005,
}

DEFAULT_EVENT_TEXTS: Dict[str, str] = {
    "bar": "识别到限高杆",
    "obstacle": "识别到无法跨越障碍",
    "cola": "识别到可乐瓶",
    "orange_ball": "识别到橙色小球",
    "football": "识别到足球",
    "blue_ball": "识别到橙色小球",
    "white_ball": "识别到足球",
}

DEFAULT_LOCAL_FILES: Dict[str, str] = {
    "bar": "bar.wav",
    "obstacle": "obstacle.wav",
    "cola": "cola.wav",
    "orange_ball": "orange_ball.wav",
    "football": "football.wav",
    "blue_ball": "orange_ball.wav",
    "white_ball": "football.wav",
}


class _PlaybackJob:
    """Small immutable-by-convention queue item (Python 3.6 compatible)."""

    def __init__(self, event_id: str, voice_key: str) -> None:
        self.event_id = event_id
        self.voice_key = voice_key


class CyberdogVoicePlayer:
    """Queue voice events without blocking the robot control loop.

    Args:
        node: Existing ``rclpy.node.Node``. Required by both ROS backends.
        backend: ``auto``, ``ros_offline``, ``ros_online``, or ``local``.
        voice_dir: Directory containing WAV files for the simulation backend.
        topic: Relative topic name. Keep it relative so the robot namespace can
            be supplied with ``--ros-args -r __ns:=/<mi_xxx>``.
        min_gap_s: Minimum spacing between accepted jobs.
        message_type: Test hook for a fake ``AudioPlayExtend`` class.
    """

    VALID_BACKENDS = {"auto", "ros_offline", "ros_online", "local"}

    def __init__(
        self,
        node: Optional[Any] = None,
        backend: str = "auto",
        voice_dir: str = "/home/cyberdog_sim/voice",
        topic: str = "speech_play_extend",
        module_name: Optional[str] = None,
        event_play_ids: Optional[Mapping[str, int]] = None,
        event_texts: Optional[Mapping[str, str]] = None,
        local_files: Optional[Mapping[str, str]] = None,
        min_gap_s: float = 0.25,
        queue_size: int = 16,
        local_timeout_s: float = 15.0,
        message_type: Optional[Type[Any]] = None,
    ) -> None:
        if backend not in self.VALID_BACKENDS:
            raise ValueError("unsupported voice backend: %s" % backend)
        if queue_size < 1:
            raise ValueError("queue_size must be positive")

        self.node = node
        self.voice_dir = voice_dir
        self.topic = topic
        self.module_name = module_name or self._node_name(node)
        self.event_play_ids = dict(DEFAULT_EVENT_PLAY_IDS)
        self.event_texts = dict(DEFAULT_EVENT_TEXTS)
        self.local_files = dict(DEFAULT_LOCAL_FILES)
        if event_play_ids:
            self.event_play_ids.update(event_play_ids)
        if event_texts:
            self.event_texts.update(event_texts)
        if local_files:
            self.local_files.update(local_files)

        self.min_gap_s = max(0.0, float(min_gap_s))
        self.local_timeout_s = max(0.1, float(local_timeout_s))
        self._lock = threading.Lock()
        self._pending: Set[str] = set()
        self._spoken: Set[str] = set()
        self._last_error: Optional[str] = None
        self._last_finished_at = 0.0
        self._compat_counter = 0
        self._busy = False
        self._closed = False
        self._queue = queue.Queue(maxsize=queue_size)
        self._publisher = None
        self._message_type = message_type

        self.backend = self._select_backend(backend)
        if self.backend.startswith("ros_"):
            if self.node is None:
                raise ValueError("node is required for ROS voice playback")
            if self._message_type is None:
                self._message_type = self._load_audio_message_type()
            self._publisher = self.node.create_publisher(self._message_type, self.topic, 2)

        self._worker = threading.Thread(
            target=self._worker_loop,
            name="cyberdog-voice-player",
            daemon=True,
        )
        self._worker.start()
        self._log("info", "voice backend=%s topic=%s" % (self.backend, self.topic))

    @staticmethod
    def _node_name(node: Optional[Any]) -> str:
        if node is not None and hasattr(node, "get_name"):
            try:
                return str(node.get_name())
            except Exception:
                pass
        return "competition_voice"

    @staticmethod
    def _load_audio_message_type() -> Type[Any]:
        try:
            from protocol.msg import AudioPlayExtend  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "protocol.msg.AudioPlayExtend is unavailable; source the CyberDog ROS2 "
                "environment or select backend='local' for simulation"
            ) from exc
        return AudioPlayExtend

    def _select_backend(self, requested: str) -> str:
        if requested != "auto":
            return requested
        if self.node is not None:
            try:
                if self._message_type is None:
                    self._message_type = self._load_audio_message_type()
                return "ros_offline"
            except RuntimeError:
                pass
        return "local"

    def _log(self, level: str, message: str) -> None:
        if self.node is not None and hasattr(self.node, "get_logger"):
            try:
                logger = self.node.get_logger()
                getattr(logger, level)("[VOICE] " + message)
                return
            except Exception:
                pass
        print("[VOICE] %s" % message)

    @property
    def last_error(self) -> Optional[str]:
        with self._lock:
            return self._last_error

    @property
    def spoken_events(self) -> Set[str]:
        with self._lock:
            return set(self._spoken)

    def speak_once(self, event_id: str, voice_key: str) -> bool:
        """Queue one event, returning ``True`` only when it was accepted.

        Duplicate events are rejected while pending and after successful
        dispatch. Failed dispatches are not marked spoken, so callers may retry.
        """
        if not event_id:
            self._set_error("empty event_id")
            return False
        if not self._known_voice_key(voice_key):
            self._set_error("unknown voice key: %s" % voice_key)
            return False

        with self._lock:
            if self._closed:
                self._last_error = "voice player is closed"
                return False
            if event_id in self._pending or event_id in self._spoken:
                return False
            self._pending.add(event_id)

        try:
            self._queue.put_nowait(_PlaybackJob(event_id=event_id, voice_key=voice_key))
        except queue.Full:
            with self._lock:
                self._pending.discard(event_id)
                self._last_error = "voice queue is full"
            self._log("warning", "queue full; event=%s" % event_id)
            return False
        return True

    def play_async(self, voice_key: str) -> bool:
        """Compatibility wrapper for the old ``VoicePlayer`` API.

        Prefer ``speak_once(event_id, voice_key)`` in new integration code.
        """
        with self._lock:
            self._compat_counter += 1
            event_id = "compat:%s:%d" % (voice_key, self._compat_counter)
        return self.speak_once(event_id, voice_key)

    def reset_event(self, event_id: str) -> None:
        with self._lock:
            self._spoken.discard(event_id)

    def is_playing(self) -> bool:
        """Return whether a voice event is queued or currently dispatching."""
        with self._lock:
            return bool(self._busy or self._pending)

    def flush(self, timeout_s: float = 20.0) -> bool:
        """Wait until queued jobs finish; intended for shutdown and smoke tests."""
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        while time.monotonic() < deadline:
            if self._queue.unfinished_tasks == 0:
                return True
            time.sleep(0.01)
        return self._queue.unfinished_tasks == 0

    def close(self, timeout_s: float = 2.0) -> None:
        """Stop accepting events and discard jobs that have not started."""
        with self._lock:
            if self._closed:
                return
            self._closed = True

        while True:
            try:
                job = self._queue.get_nowait()
            except queue.Empty:
                break
            if job is not None:
                with self._lock:
                    self._pending.discard(job.event_id)
            self._queue.task_done()

        try:
            self._queue.put_nowait(None)
        except queue.Full:
            return
        self._worker.join(timeout=max(0.0, float(timeout_s)))

    def _known_voice_key(self, voice_key: str) -> bool:
        if self.backend == "ros_offline":
            return voice_key in self.event_play_ids
        if self.backend == "ros_online":
            return voice_key in self.event_texts
        return voice_key in self.local_files

    def _worker_loop(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:
                self._queue.task_done()
                return

            with self._lock:
                discard = self._closed
                if discard:
                    self._pending.discard(job.event_id)
                else:
                    self._busy = True
            if discard:
                self._queue.task_done()
                continue
            try:
                remaining = self.min_gap_s - (time.monotonic() - self._last_finished_at)
                if remaining > 0:
                    time.sleep(remaining)
                with self._lock:
                    discard = self._closed
                if discard:
                    ok, error = False, 'voice player closed before dispatch'
                else:
                    ok, error = self._dispatch(job.voice_key)
            finally:
                with self._lock:
                    self._busy = False
            with self._lock:
                self._pending.discard(job.event_id)
                if ok:
                    self._spoken.add(job.event_id)
                    self._last_error = None
                else:
                    self._last_error = error
            self._last_finished_at = time.monotonic()
            if ok:
                self._log("info", "queued event=%s key=%s" % (job.event_id, job.voice_key))
            else:
                self._log("warning", "failed event=%s: %s" % (job.event_id, error))
            self._queue.task_done()

    def _dispatch(self, voice_key: str) -> Tuple[bool, str]:
        if self.backend == "ros_offline":
            return self._publish_ros(voice_key, online=False)
        if self.backend == "ros_online":
            return self._publish_ros(voice_key, online=True)
        return self._play_local(voice_key)

    def _publish_ros(self, voice_key: str, online: bool) -> Tuple[bool, str]:
        try:
            msg = self._message_type()
            msg.module_name = self.module_name
            msg.is_online = bool(online)
            msg.speech.module_name = self.module_name
            msg.speech.play_id = (
                0 if online else int(self.event_play_ids.get(voice_key, 0)))
            msg.text = self.event_texts.get(voice_key, "") if online else ""
            self._publisher.publish(msg)
            return True, ""
        except Exception as exc:
            return False, "ROS publish failed: %s" % exc

    def _play_local(self, voice_key: str) -> Tuple[bool, str]:
        player = shutil.which("aplay")
        if not player:
            return False, "aplay is not installed"
        path = os.path.join(self.voice_dir, self.local_files[voice_key])
        if not os.path.isfile(path):
            return False, "audio file not found: %s" % path
        try:
            result = subprocess.run(
                [player, "-q", path],
                check=False,
                timeout=self.local_timeout_s,
            )
        except subprocess.TimeoutExpired:
            return False, "local playback timed out: %s" % path
        except OSError as exc:
            return False, "local playback failed: %s" % exc
        if result.returncode != 0:
            return False, "aplay exited with code %d" % result.returncode
        return True, ""

    def _set_error(self, message: str) -> None:
        with self._lock:
            self._last_error = message
        self._log("warning", message)
