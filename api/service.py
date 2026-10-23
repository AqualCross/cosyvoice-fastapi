"""Engine wrapper around the upstream CosyVoice package.

The upstream repository is treated as an external dependency: it is imported
from its clone directory (``COSYVOICE_REPO``) and no upstream source is copied
into this project. Also loads the Matcha-TTS submodule it needs.
"""

import gc
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

AUDIO_EXTS = (".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".webm", ".opus")
VOICE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
MAX_VOICE_BYTES = 20 * 1024 * 1024
MAX_PROMPT_SECONDS = 28.0
MIN_PROMPT_SECONDS = 1.0
PCM16_MAX = 32767.0
CV3_PROMPT_PREFIX = "You are a helpful assistant.<|endofprompt|>"


class VoiceNotFoundError(ValueError):
    """Raised when a voice id / audio path cannot be resolved."""


class ModelNotFoundError(ValueError):
    """Raised when a requested model id is not configured."""


class TTSNotReadyError(RuntimeError):
    """Raised when the model is still loading."""


@dataclass
class LoadedModel:
    key: str
    path: str
    engine: object
    sample_rate: int
    text_frontend: str
    needs_prompt_prefix: bool
    load_seconds: float
    last_used: float
    spk_ids: Dict[str, str] = field(default_factory=dict)


class CosyVoiceService:
    """Loads CosyVoice2/3 on demand and serializes GPU inference behind a lock."""

    def __init__(
        self,
        repo_dir: str,
        model_dirs: Dict[str, str],
        voices_dir: str = "voices",
        wetext_dir: Optional[str] = None,
        default_model: str = "cosyvoice2",
        fp16: bool = False,
        max_loaded: int = 1,
    ):
        self.repo_dir = os.path.abspath(repo_dir)
        self.model_dirs = {k: os.path.abspath(v) for k, v in model_dirs.items()}
        self.voices_dir = voices_dir
        self.wetext_dir = wetext_dir or os.path.join(self.repo_dir, "pretrained_models", "wetext")
        self.default_model = default_model
        self.fp16 = fp16
        self.max_loaded = max(1, max_loaded)

        os.makedirs(self.voices_dir, exist_ok=True)

        self._slots: Dict[str, LoadedModel] = {}
        self._lock = threading.Lock()  # one inference at a time on the GPU
        self._load_lock = threading.Lock()
        self._warned_voices = set()

    # ------------------------------------------------------------------ loading

    @property
    def configured_models(self) -> List[str]:
        return list(self.model_dirs)

    def loaded_models(self) -> List[str]:
        return list(self._slots)

    @property
    def ready(self) -> bool:
        return self.default_model in self._slots

    def _ensure_upstream(self) -> None:
        if not os.path.isdir(os.path.join(self.repo_dir, "cosyvoice")):
            raise RuntimeError(
                f"CosyVoice repository not found at {self.repo_dir}; "
                "set COSYVOICE_REPO or pass --repo"
            )
        for entry in (self.repo_dir, os.path.join(self.repo_dir, "third_party", "Matcha-TTS")):
            if entry not in sys.path:
                sys.path.insert(0, entry)

    @contextmanager
    def _local_wetext(self):
        """Point wetext's FST lookup at the local directory during model construction.

        The upstream frontend instantiates ``wetext.Normalizer()``, which calls
        ``snapshot_download('pengzhendong/wetext')`` and hits ModelScope at every
        model load (and fails behind rate limiting). The four FST files are
        already on disk, so serve them locally instead.
        """
        zh_dir = os.path.join(self.wetext_dir, "zh", "tn")
        en_dir = os.path.join(self.wetext_dir, "en", "tn")
        if not (os.path.isdir(zh_dir) and os.path.isdir(en_dir)):
            yield
            return

        try:
            import wetext.wetext as wetext_module
        except ImportError:
            yield
            return

        original = wetext_module.snapshot_download

        def local_snapshot_download(repo_id, *args, **kwargs):
            if repo_id == "pengzhendong/wetext":
                return self.wetext_dir
            return original(repo_id, *args, **kwargs)

        wetext_module.snapshot_download = local_snapshot_download
        try:
            yield
        finally:
            wetext_module.snapshot_download = original

    def load(self, model: Optional[str] = None) -> LoadedModel:
        key = model or self.default_model
        slot = self._slots.get(key)
        if slot is not None:
            slot.last_used = time.time()
            return slot

        if key not in self.model_dirs:
            raise ModelNotFoundError(
                f"unknown model '{key}', configured: {', '.join(self.configured_models)}"
            )

        with self._load_lock:
            slot = self._slots.get(key)
            if slot is not None:
                slot.last_used = time.time()
                return slot

            model_path = self.model_dirs[key]
            if not os.path.isdir(model_path):
                raise RuntimeError(f"model directory not found: {model_path}")
            self._ensure_upstream()

            while len(self._slots) >= self.max_loaded:
                self._evict_oldest()

            from cosyvoice.cli.cosyvoice import AutoModel

            started = time.perf_counter()
            with self._local_wetext():
                engine = AutoModel(model_dir=model_path, fp16=self.fp16)
            slot = LoadedModel(
                key=key,
                path=model_path,
                engine=engine,
                sample_rate=int(engine.sample_rate),
                text_frontend=getattr(engine.frontend, "text_frontend", "") or "none",
                needs_prompt_prefix=type(engine).__name__ == "CosyVoice3",
                load_seconds=time.perf_counter() - started,
                last_used=time.time(),
            )
            slot.text_frontend = self._enable_text_frontend(slot)
            self._slots[key] = slot
            logger.info(
                "model loaded: %s (%s) in %.1fs | sample_rate=%d fp16=%s text_frontend=%s",
                key, model_path, slot.load_seconds, slot.sample_rate, self.fp16, slot.text_frontend,
            )
            return slot

    def _enable_text_frontend(self, slot: LoadedModel) -> str:
        """Attach the wetext normalizers from a local FST directory.

        The wetext package downloads its FSTs from ModelScope at import time,
        which fails behind rate limiting; the files are kept on disk instead.
        """
        frontend = slot.engine.frontend
        if getattr(frontend, "text_frontend", "") == "wetext":
            return "wetext"

        zh_dir = os.path.join(self.wetext_dir, "zh", "tn")
        en_dir = os.path.join(self.wetext_dir, "en", "tn")
        if not (os.path.isdir(zh_dir) and os.path.isdir(en_dir)):
            logger.warning(
                "text frontend disabled: wetext resources missing at %s", self.wetext_dir
            )
            return getattr(frontend, "text_frontend", "") or "none"

        from wetext import Normalizer

        frontend.zh_tn_model = Normalizer(
            tagger_path=os.path.join(zh_dir, "tagger.fst"),
            verbalizer_path=os.path.join(zh_dir, "verbalizer.fst"),
            lang="zh",
        )
        frontend.en_tn_model = Normalizer(
            tagger_path=os.path.join(en_dir, "tagger.fst"),
            verbalizer_path=os.path.join(en_dir, "verbalizer.fst"),
            lang="en",
        )
        frontend.text_frontend = "wetext"
        return "wetext"

    def _evict_oldest(self) -> None:
        if not self._slots:
            return
        key = min(self._slots, key=lambda k: self._slots[k].last_used)
        slot = self._slots.pop(key)
        logger.info("unloading model %s to free VRAM", key)
        del slot.engine
        slot.engine = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    def unload_all(self) -> None:
        with self._load_lock:
            for key in list(self._slots):
                self._evict_oldest()

    def warmup(self, model: Optional[str] = None, text: str = "你好，欢迎使用语音合成服务。") -> None:
        slot = self.load(model)
        asset = os.path.join(self.repo_dir, "asset", "zero_shot_prompt.wav")
        transcript = "希望你以后能够做的比我还好呦。"
        if not os.path.isfile(asset):
            logger.warning("warmup skipped: %s not found", asset)
            return
        started = time.perf_counter()
        try:
            self.synthesize(
                model=slot.key, text=text, voice=asset, prompt_text=transcript,
                speed=1.0, text_frontend=True,
            )
            logger.info("warmup done in %.1fs", time.perf_counter() - started)
        except Exception:  # noqa: BLE001 - warmup must never kill the server
            logger.exception("warmup inference failed; continuing anyway")

    # ------------------------------------------------------------------ voices

    def list_voices(self) -> List[dict]:
        voices = []
        if not os.path.isdir(self.voices_dir):
            return voices
        for name in sorted(os.listdir(self.voices_dir)):
            base, ext = os.path.splitext(name)
            if ext.lower() not in AUDIO_EXTS:
                continue
            meta = self._read_meta(base)
            voices.append({
                "id": base,
                "path": os.path.join(self.voices_dir, name),
                "duration_seconds": meta.get("duration_seconds"),
                "prompt_text": meta.get("prompt_text", ""),
                "has_prompt_text": bool(meta.get("prompt_text")),
            })
        return voices

    def _meta_path(self, voice_id: str) -> str:
        return os.path.join(self.voices_dir, voice_id + ".json")

    def _read_meta(self, voice_id: str) -> dict:
        path = self._meta_path(voice_id)
        if not os.path.isfile(path):
            return {}
        try:
            with open(path, encoding="utf-8") as fp:
                return json.load(fp)
        except (OSError, ValueError):
            logger.warning("unreadable voice metadata: %s", path)
            return {}

    def resolve_voice(self, voice: str) -> Tuple[str, str]:
        """Map a voice id (or audio path) to ``(audio_path, stored_prompt_text)``."""
        if not voice:
            raise VoiceNotFoundError("voice must not be empty")
        if ":" in voice:
            voice = voice.split(":", 1)[1]

        if os.path.isfile(voice) and os.path.splitext(voice)[1].lower() in AUDIO_EXTS:
            return os.path.abspath(voice), ""

        for ext in AUDIO_EXTS:
            candidate = os.path.join(self.voices_dir, voice + ext)
            if os.path.isfile(candidate):
                meta = self._read_meta(voice if ext == ".wav" else os.path.splitext(voice)[0])
                return os.path.abspath(candidate), meta.get("prompt_text", "")

        candidate = os.path.join(self.voices_dir, voice)
        if os.path.isfile(candidate):
            base = os.path.splitext(os.path.basename(voice))[0]
            return os.path.abspath(candidate), self._read_meta(base).get("prompt_text", "")

        available = ", ".join(v["id"] for v in self.list_voices()) or "<none>"
        raise VoiceNotFoundError(f"Voice '{voice}' not found. Available voices: {available}")

    def save_voice(self, name: str, data: bytes, ext: str, prompt_text: str = "") -> dict:
        """Store an uploaded reference as mono 16-bit WAV plus a JSON sidecar."""
        if not VOICE_NAME_PATTERN.match(name or ""):
            raise ValueError(
                "voice name must be 1-64 chars of letters, digits, '_', '-' or '.' "
                "(it is used as a file name)"
            )
        ext = (ext or "").lower()
        if ext not in AUDIO_EXTS:
            raise ValueError(f"unsupported audio extension '{ext}'")
        if not data:
            raise ValueError("uploaded file is empty")
        if len(data) > MAX_VOICE_BYTES:
            raise ValueError(f"audio file too large ({len(data) / 1024 ** 2:.1f} MiB > 20 MiB)")

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg is required to import reference audio but was not found in PATH")

        with tempfile.TemporaryDirectory(prefix="cosyvoice_voice_") as tmp:
            src = os.path.join(tmp, "in" + ext)
            with open(src, "wb") as fp:
                fp.write(data)
            dst = os.path.join(tmp, "out.wav")
            proc = subprocess.run(
                [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", src,
                 "-ac", "1", "-c:a", "pcm_s16le", dst],
                capture_output=True,
            )
            if proc.returncode != 0:
                raise ValueError(
                    "uploaded audio is not decodable: "
                    + proc.stderr.decode(errors="replace").strip()[:300]
                )
            duration = self.probe_duration(dst)
            if duration < MIN_PROMPT_SECONDS:
                raise ValueError(f"reference audio too short ({duration:.2f}s < {MIN_PROMPT_SECONDS}s)")
            if duration > MAX_PROMPT_SECONDS:
                raise ValueError(
                    f"reference audio too long ({duration:.2f}s > {MAX_PROMPT_SECONDS}s); "
                    "CosyVoice only accepts prompts up to 30s"
                )
            target = os.path.join(self.voices_dir, name + ".wav")
            part = target + ".part"
            shutil.copyfile(dst, part)
            os.replace(part, target)

        for other_ext in AUDIO_EXTS:
            if other_ext == ".wav":
                continue
            existing = os.path.join(self.voices_dir, name + other_ext)
            if os.path.isfile(existing):
                os.remove(existing)

        prompt_text = (prompt_text or "").strip()
        with open(self._meta_path(name), "w", encoding="utf-8") as fp:
            json.dump({
                "id": name,
                "prompt_text": prompt_text,
                "duration_seconds": round(duration, 2),
                "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }, fp, ensure_ascii=False, indent=2)

        logger.info("voice saved: %s (%.2fs, transcript=%s)", target, duration, bool(prompt_text))
        return {
            "id": name,
            "path": target,
            "duration_seconds": round(duration, 2),
            "has_prompt_text": bool(prompt_text),
        }

    def delete_voice(self, voice_id: str) -> List[str]:
        if not VOICE_NAME_PATTERN.match(voice_id or ""):
            raise ValueError("invalid voice id")
        removed = []
        for ext in AUDIO_EXTS:
            path = os.path.join(self.voices_dir, voice_id + ext)
            if os.path.isfile(path):
                os.remove(path)
                removed.append(path)
        meta = self._meta_path(voice_id)
        if os.path.isfile(meta):
            os.remove(meta)
            removed.append(meta)
        if not removed:
            raise VoiceNotFoundError(f"Voice '{voice_id}' not found")
        logger.info("voice deleted: %s", ", ".join(removed))
        return removed

    @staticmethod
    def probe_duration(path: str) -> float:
        ffprobe = shutil.which("ffprobe")
        if ffprobe is None:
            raise RuntimeError("ffprobe not found in PATH")
        proc = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=30,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or "ffprobe failed")
        try:
            return float(proc.stdout.strip())
        except ValueError as exc:
            raise RuntimeError("no audio stream found") from exc

    # ------------------------------------------------------------------ inference

    def _ensure_spk(self, slot: LoadedModel, voice_path: str, prompt_text: str) -> str:
        prompt = (prompt_text or "").strip()
        if slot.needs_prompt_prefix and prompt and CV3_PROMPT_PREFIX not in prompt:
            prompt = CV3_PROMPT_PREFIX + prompt
        cache_key = f"{voice_path}|{os.path.getmtime(voice_path):.0f}|{prompt}"
        cached = slot.spk_ids.get(cache_key)
        if cached:
            return cached

        if not prompt:
            warning_key = (slot.key, voice_path)
            if warning_key not in self._warned_voices:
                self._warned_voices.add(warning_key)
                logger.warning(
                    "voice %s has no transcript; CosyVoice zero-shot cloning will be weaker "
                    "— upload the reference text with the voice or pass prompt_text",
                    os.path.basename(voice_path),
                )
        else:
            prompt = slot.engine.frontend.text_normalize(prompt, split=False, text_frontend=True)

        spk_id = "api_" + hashlib.md5(cache_key.encode("utf-8")).hexdigest()[:12]
        slot.engine.add_zero_shot_spk(prompt, voice_path, spk_id)
        slot.spk_ids[cache_key] = spk_id
        return spk_id

    @staticmethod
    def _to_int16(tensor) -> np.ndarray:
        """Convert a (1, N) float waveform in [-1, 1] to int16 numpy."""
        import torch

        arr = tensor.detach().to(device="cpu", dtype=torch.float32).numpy()
        arr = np.clip(arr, -1.0, 1.0) * PCM16_MAX
        if arr.ndim == 2:
            arr = arr[0]
        return arr.astype(np.int16)

    @staticmethod
    def _seed_rng(seed: Optional[int]) -> None:
        if seed is None:
            return
        from cosyvoice.utils.common import set_all_random_seed

        set_all_random_seed(int(seed))

    def synthesize(
        self,
        model: str,
        text: str,
        voice: str,
        speed: float = 1.0,
        text_frontend: bool = True,
        prompt_text: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Tuple[int, np.ndarray]:
        slot = self.load(model)
        voice_path, stored_prompt = self.resolve_voice(voice)
        prompt = stored_prompt if prompt_text is None else prompt_text

        with self._lock:
            self._seed_rng(seed)
            spk_id = self._ensure_spk(slot, voice_path, prompt)
            outputs = list(slot.engine.inference_zero_shot(
                text, "", "", zero_shot_spk_id=spk_id,
                stream=False, speed=speed, text_frontend=text_frontend,
            ))

        parts = [self._to_int16(item["tts_speech"]) for item in outputs]
        samples = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int16)
        if samples.size == 0:
            raise RuntimeError("CosyVoice returned no audio")
        return slot.sample_rate, samples

    def synthesize_stream(
        self,
        model: str,
        text: str,
        voice: str,
        speed: float = 1.0,
        text_frontend: bool = True,
        prompt_text: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Iterator[Tuple[int, np.ndarray]]:
        """Yield ``(sample_rate, int16 mono)`` as each text segment is generated."""
        slot = self.load(model)
        voice_path, stored_prompt = self.resolve_voice(voice)
        prompt = stored_prompt if prompt_text is None else prompt_text

        with self._lock:
            self._seed_rng(seed)
            spk_id = self._ensure_spk(slot, voice_path, prompt)
            for item in slot.engine.inference_zero_shot(
                text, "", "", zero_shot_spk_id=spk_id,
                stream=True, speed=speed, text_frontend=text_frontend,
            ):
                yield slot.sample_rate, self._to_int16(item["tts_speech"])

    # ------------------------------------------------------------------ encoding

    @staticmethod
    def write_pcm_wav(path: str, samples: np.ndarray, sample_rate: int) -> None:
        with wave.open(path, "wb") as fp:
            fp.setnchannels(1)
            fp.setsampwidth(2)
            fp.setframerate(sample_rate)
            fp.writeframes(samples.astype(np.int16).tobytes())

    def encode(
        self,
        samples: np.ndarray,
        sample_rate: int,
        response_format: str = "wav",
        target_sample_rate: Optional[int] = None,
        gain_db: float = 0.0,
    ) -> bytes:
        target_sample_rate = target_sample_rate or sample_rate
        response_format = response_format.lower()

        if response_format == "pcm":
            return self._apply_gain(samples, gain_db).tobytes()

        if (
            response_format == "wav"
            and target_sample_rate == sample_rate
            and abs(gain_db) < 1e-6
        ):
            import io

            buf = io.BytesIO()
            with wave.open(buf, "wb") as fp:
                fp.setnchannels(1)
                fp.setsampwidth(2)
                fp.setframerate(sample_rate)
                fp.writeframes(samples.astype(np.int16).tobytes())
            return buf.getvalue()

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError(
                "ffmpeg is required for gain/resample/mp3/ogg conversion but was not found in PATH"
            )

        with tempfile.TemporaryDirectory(prefix="cosyvoice_api_") as tmp:
            in_path = os.path.join(tmp, "in.wav")
            out_path = os.path.join(tmp, f"out.{response_format}")
            self.write_pcm_wav(in_path, samples, sample_rate)

            cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", in_path]
            if abs(gain_db) > 1e-6:
                cmd += ["-af", f"volume={gain_db}dB"]
            cmd += ["-ar", str(target_sample_rate)]
            if response_format == "mp3":
                cmd += ["-c:a", "libmp3lame", "-b:a", "192k"]
            elif response_format == "ogg":
                cmd += ["-c:a", "libvorbis", "-q:a", "5"]
            elif response_format == "wav":
                cmd += ["-c:a", "pcm_s16le"]
            else:
                raise ValueError(f"Unsupported response_format: {response_format}")
            cmd.append(out_path)

            proc = subprocess.run(cmd, capture_output=True)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"ffmpeg failed ({proc.returncode}): {proc.stderr.decode(errors='replace')}"
                )
            with open(out_path, "rb") as fp:
                return fp.read()

    @staticmethod
    def _apply_gain(samples: np.ndarray, gain_db: float) -> np.ndarray:
        if abs(gain_db) < 1e-6:
            return samples.astype(np.int16, copy=False)
        out = samples.astype(np.float32) * (10 ** (gain_db / 20.0))
        return np.clip(out, -PCM16_MAX, PCM16_MAX).astype(np.int16)

    def iter_encoded_stream(
        self,
        chunks: Iterator[Tuple[int, np.ndarray]],
        sample_rate: int,
        response_format: str = "wav",
        target_sample_rate: Optional[int] = None,
        gain_db: float = 0.0,
    ) -> Iterator[bytes]:
        fmt = response_format.lower()

        if fmt == "pcm":
            for _sr, samples in chunks:
                yield self._apply_gain(samples, gain_db).tobytes()
            return

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg is required for streaming mp3/ogg/wav but was not found in PATH")

        target_sample_rate = target_sample_rate or sample_rate
        cmd = [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-f", "s16le", "-ar", str(sample_rate), "-ac", "1", "-i", "pipe:0",
        ]
        if abs(gain_db) > 1e-6:
            cmd += ["-af", f"volume={gain_db}dB"]
        cmd += ["-ar", str(target_sample_rate)]
        if fmt == "mp3":
            cmd += ["-c:a", "libmp3lame", "-b:a", "192k", "-f", "mp3"]
        elif fmt == "ogg":
            cmd += ["-c:a", "libvorbis", "-q:a", "5", "-f", "ogg"]
        elif fmt == "wav":
            cmd += ["-c:a", "pcm_s16le", "-f", "wav"]
        else:
            raise ValueError(f"Unsupported response_format: {response_format}")
        cmd.append("pipe:1")

        proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            bufsize=0,
        )

        stderr_lines: List[bytes] = []

        def _drain_stderr() -> None:
            try:
                for line in iter(proc.stderr.readline, b""):
                    stderr_lines.append(line)
            except Exception:  # noqa: BLE001
                pass

        def _feed_stdin() -> None:
            try:
                for _sr, samples in chunks:
                    proc.stdin.write(samples.astype(np.int16, copy=False).tobytes())
                proc.stdin.close()
            except Exception:  # noqa: BLE001 - broken pipe on client disconnect
                try:
                    chunks.close()  # type: ignore[attr-defined] - releases the model lock
                except Exception:  # noqa: BLE001
                    pass
                try:
                    proc.stdin.close()
                except Exception:  # noqa: BLE001
                    pass

        threading.Thread(target=_drain_stderr, daemon=True).start()
        feeder = threading.Thread(target=_feed_stdin, daemon=True)
        feeder.start()

        try:
            while True:
                data = proc.stdout.read(64 * 1024)
                if not data:
                    break
                yield data
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
            feeder.join(timeout=30)
            if proc.returncode not in (0, None) and stderr_lines:
                logger.warning(
                    "ffmpeg stream exited with code %s: %s",
                    proc.returncode,
                    b"".join(stderr_lines[-5:]).decode(errors="replace").strip(),
                )

    # ------------------------------------------------------------------ reporting

    def model_report(self) -> List[dict]:
        report = []
        for key, path in self.model_dirs.items():
            slot = self._slots.get(key)
            entry = {
                "id": key,
                "path": path,
                "loaded": slot is not None,
                "exists": os.path.isdir(path),
            }
            if slot is not None:
                entry.update({
                    "sample_rate": slot.sample_rate,
                    "text_frontend": slot.text_frontend,
                    "fp16": self.fp16,
                    "load_seconds": round(slot.load_seconds, 1),
                    "needs_prompt_prefix": slot.needs_prompt_prefix,
                })
            report.append(entry)
        return report

    def vram_report(self) -> Optional[dict]:
        try:
            import torch

            if not torch.cuda.is_available():
                return None
            idx = torch.cuda.current_device()
            to_gib = lambda n: round(n / 1024 ** 3, 2)  # noqa: E731
            free, total = torch.cuda.mem_get_info(idx)
            return {
                "allocated_gib": to_gib(torch.cuda.memory_allocated(idx)),
                "reserved_gib": to_gib(torch.cuda.memory_reserved(idx)),
                "peak_allocated_gib": to_gib(torch.cuda.max_memory_allocated(idx)),
                "peak_reserved_gib": to_gib(torch.cuda.max_memory_reserved(idx)),
                "device_free_gib": to_gib(free),
                "device_total_gib": to_gib(total),
            }
        except Exception:  # noqa: BLE001
            return None
