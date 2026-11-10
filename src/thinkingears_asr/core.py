import hashlib
import json
import math
import os
import re
import shutil
import threading
import time
import traceback
from pathlib import Path

from ._helpers import choose_input_device, sanitize_display_text
from .lexicon import builtin_lexicon, load_lexicon

def tempfile_dir():
    return os.environ.get("TEMP") or os.environ.get("TMP") or str(Path.home())


class ASREnhancer:
    """Enhance ASR text, with optional local Paraformer and microphone support."""

    def __init__(
        self, root=None, *, model_dir=None, hotwords=None, corrections=None,
        lexicon_path=None, use_builtin_lexicon=True,
    ):
        self.root = Path(root) if root is not None else Path.cwd()
        self.model_dir = (
            Path(model_dir) if model_dir is not None
            else self.root / "models" / "sherpa-onnx-paraformer-zh-2023-09-14"
        )
        self.master_path = Path(lexicon_path) if lexicon_path is not None else None
        self.use_builtin_lexicon = use_builtin_lexicon
        self.sample_rate = 16000
        self.max_seconds = 10
        self.short_pause_ms = 180
        self.long_pause_ms = 400
        self._pause_text_parts = []
        self._pause_audio_parts = []
        self.threshold_level = 5
        self.energy_threshold = self._threshold_from_level(self.threshold_level)
        self.noise_floor = self.energy_threshold
        self.noise_margin = 1.0
        self.lock = threading.RLock()
        self._initialization_lock = threading.Lock()
        self._recognition_lock = threading.Lock()
        self.initialized = False
        self.loading = False
        self.error = ""
        self.recognizer = None
        self.hotwords = []
        self.hotword_pinyin = {}
        self.master_words = set()
        self.master_ready = False
        self.active = False
        self.final = False
        self.started_at = 0.0
        self.level = 0.0
        self.raw_text = ""
        self.text = ""
        self.committed_text = ""
        self.segment_chunks = []
        self._silence_start = None
        self._speech_detected = False
        self.voice_metrics = self.empty_voice_metrics()
        self.punct_boundaries = []
        self.hint = ""
        self.audio_chunks = []
        self.pre_roll_chunks = []
        self.pre_roll_max_blocks = 12
        self.worker = None
        self.input_device_index = None
        self.input_device_name = ""
        self.last_decode = 0.0
        from pypinyin import lazy_pinyin
        self._pinyin = lazy_pinyin
        self.freq_map = dict(corrections or {})
        if any(
            not isinstance(src, str) or not src or not isinstance(dst, str)
            for src, dst in self.freq_map.items()
        ):
            raise ValueError("Corrections must map nonempty strings to strings.")
        self.set_hotwords(hotwords or [])
        self._load_master_dict()

    @property
    def lexicon_count(self):
        return len(self.master_words)

    def enhance(self, text, *, final=False):
        """Correct text; opt into heuristic punctuation with final=True."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        with self.lock:
            return self.correct(text, final=final)

    def analyze(self, audio, text=""):
        """Measure mono float audio sampled at 16 kHz."""
        audio = self._waveform(audio)
        return self.analyze_physical(audio, text)

    @staticmethod
    def _waveform(audio):
        import numpy as np
        audio = np.asarray(audio)
        if np.iscomplexobj(audio):
            raise ValueError("audio must contain real samples")
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 1 or not np.isfinite(audio).all():
            raise ValueError("audio must be a finite mono waveform at 16000 Hz")
        return audio

    def initialize(self):
        """Synchronously initialize the optional local ASR backend."""
        self._init()
        with self.lock:
            if not self.initialized:
                raise RuntimeError(self.error or "ASR initialization failed")
        return self

    def transcribe(self, audio):
        """Recognize a mono, float32, 16 kHz waveform using local Paraformer."""
        import numpy as np
        audio = self._waveform(audio)
        rms = float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0
        if rms < self.energy_threshold:
            return {
                "raw_text": "",
                "text": "",
                "voice_metrics": self.analyze_physical(audio, ""),
            }
        self.initialize()
        with self.lock:
            raw = self._recognize(audio)
            return {
                "raw_text": raw,
                "text": self.correct(raw, final=True, audio=audio),
                "voice_metrics": self.analyze_physical(audio, raw),
            }

    def _threshold_from_level(self, level):
        level = max(1, min(10, int(level)))
        return round(0.0085 + (level - 1) * 0.002, 4)

    def set_threshold_level(self, level):
        with self.lock:
            self.threshold_level = max(1, min(10, int(level)))
            self.energy_threshold = self._threshold_from_level(self.threshold_level)
            self.noise_floor = self.energy_threshold
        return self.snapshot()

    def init_async(self):
        with self.lock:
            if self.initialized or self.loading:
                return
            self.loading = True
            self.error = ""
        threading.Thread(target=self._init, daemon=True).start()

    def _init(self):
        with self._initialization_lock:
            if self.initialized:
                return
            with self.lock:
                self.loading = True
            self._initialize_backend()

    def _initialize_backend(self):
        try:
            import sherpa_onnx
            model = self.model_dir / "model.int8.onnx"
            tokens = self.model_dir / "tokens.txt"
            if not model.is_file() or not tokens.is_file():
                raise FileNotFoundError(f"缺少 ASR 模型文件：{self.model_dir}")

            runtime_model_dir = self._prepare_runtime_model_dir(model, tokens)
            runtime_model = runtime_model_dir / "model.int8.onnx"
            runtime_tokens = runtime_model_dir / "tokens.txt"

            self.recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(
                paraformer=str(runtime_model),
                tokens=str(runtime_tokens),
                num_threads=4,
                sample_rate=self.sample_rate,
                feature_dim=80,
                provider="cpu",
            )
            with self.lock:
                self.initialized = True
                self.loading = False
                self.error = ""
        except Exception as exc:
            with self.lock:
                self.initialized = False
                self.loading = False
                self.error = f"{exc}\n{traceback.format_exc(limit=2)}"

    def _prepare_runtime_model_dir(self, model, tokens):
        """把模型准备到纯英文缓存路径，避免部分 sherpa-onnx/Windows 环境读取中文路径失败。"""
        try:
            str(model).encode("ascii")
            str(tokens).encode("ascii")
            return self.model_dir
        except UnicodeEncodeError:
            pass
        digest = hashlib.sha256()
        for path in (model, tokens):
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
        model_hash = digest.hexdigest()[:16]
        cache_root = self._ascii_cache_root() / "asr_models"
        runtime_dir = cache_root / model_hash
        runtime_model = runtime_dir / "model.int8.onnx"
        runtime_tokens = runtime_dir / "tokens.txt"

        try:
            if runtime_model.is_file() and runtime_tokens.is_file():
                if runtime_model.stat().st_size == model.stat().st_size and runtime_tokens.stat().st_size == tokens.stat().st_size:
                    return runtime_dir
            runtime_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(model, runtime_model)
            shutil.copy2(tokens, runtime_tokens)
            return runtime_dir
        except Exception:
            return self.model_dir

    def _ascii_cache_root(self):
        candidates = []
        custom = os.environ.get("THINKINGEARS_CACHE")
        if custom:
            candidates.append(Path(custom))
        public = os.environ.get("PUBLIC")
        if public:
            candidates.append(Path(public) / "ThinkingEars-ASR" / "cache")
        candidates.append(Path(tempfile_dir()) / "ThinkingEars-ASR" / "cache")

        for candidate in candidates:
            try:
                str(candidate).encode("ascii")
                candidate.mkdir(parents=True, exist_ok=True)
                return candidate
            except Exception:
                continue
        return self.root / ".asr_cache"

    def set_hotwords(self, hotwords):
        """Replace hotwords from a sequence or grouped JSON file."""
        if isinstance(hotwords, (str, Path)):
            hotwords = json.loads(Path(hotwords).read_text(encoding="utf-8-sig"))
        if isinstance(hotwords, dict):
            if any(not isinstance(group, list) for group in hotwords.values()):
                raise ValueError("Hotword groups must be lists of strings.")
            hotwords = [word for group in hotwords.values() for word in group]
        if not isinstance(hotwords, (list, tuple, set, frozenset)):
            raise ValueError("Hotwords must be a sequence or grouped JSON object.")
        if any(not isinstance(word, str) for word in hotwords):
            raise ValueError("Hotwords must be strings.")
        words = sorted(
            {word.strip() for word in hotwords if word.strip()},
            key=lambda word: (-len(word), word),
        )
        candidates = {}
        for word in words:
            if re.fullmatch(r"[\u3400-\u4dbf\u4e00-\u9fff]{2,10}", word):
                py = self._py(word)
                if py:
                    candidates.setdefault((len(word), py), set()).add(word)
        mapping = {
            py: next(iter(matches)) for py, matches in candidates.items()
            if len(matches) == 1
        }
        with self.lock:
            self.hotwords = words
            self.hotword_pinyin = mapping

    def _load_master_dict(self):
        if self.master_path is not None:
            self.master_words = load_lexicon(self.master_path)
        elif self.use_builtin_lexicon:
            self.master_words = builtin_lexicon()
        else:
            self.master_words = frozenset()
        self.master_ready = bool(self.master_words)

    def _py(self, text):
        if not self._pinyin:
            return ""
        return " ".join(self._pinyin(text, errors="ignore"))

    def start(self, seconds=10):
        self.init_async()
        with self.lock:
            if not self.initialized:
                return False, "ASR 正在加载" if self.loading else (self.error or "ASR 未初始化")
            if self.active:
                return True, "ASR 已在识别"
            if self.worker is not None and self.worker.is_alive():
                return False, "上一段 ASR 尚未处理完成"
            self.max_seconds = max(1, min(30, int(seconds or 10)))
            self.active = True
            self.final = False
            self.started_at = time.time()
            self.level = 0.0
            self.raw_text = ""
            self.text = ""
            self.hint = ""
            self.error = ""
            self.voice_metrics = self.empty_voice_metrics()
            self.audio_chunks = []
            self.segment_chunks = []
            self.pre_roll_chunks = []
            self.committed_text = ""
            self._silence_start = None
            self._speech_detected = False
            self._pause_text_parts = []
            self._pause_audio_parts = []
            self.last_decode = 0.0
            self.worker = threading.Thread(target=self._record_loop, daemon=True)
            self.worker.start()
            return True, "开始识别"

    def stop(self):
        with self.lock:
            self.active = False
            worker = self.worker
        if worker and worker is not threading.current_thread() and worker.is_alive():
            # Releasing the input stream before final decoding avoids a Bluetooth
            # hands-free profile race that can truncate the last spoken phrase.
            worker.join(timeout=1.5)
        return self.snapshot()

    def _select_input_device(self, sd):
        devices = [dict(device) for device in sd.query_devices()]
        default = getattr(sd.default, "device", None)
        try:
            default_index = int(default[0])
        except Exception:
            default_index = None
        index, name = choose_input_device(devices, default_index)
        with self.lock:
            self.input_device_index = index
            self.input_device_name = name
        return index

    def _record_loop(self):
        try:
            import sounddevice as sd
            block = int(self.sample_rate * 0.032)
            device = self._select_input_device(sd)
            with sd.InputStream(
                device=device,
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=block,
                latency="low",
            ) as stream:
                while True:
                    with self.lock:
                        active = self.active
                        elapsed = time.time() - self.started_at if self.started_at else 0
                    if not active or elapsed >= self.max_seconds:
                        break
                    audio, _ = stream.read(block)
                    audio = audio.reshape(-1).astype("float32")
                    self._handle_audio_block(audio)
                    if time.time() - self.last_decode > 0.15:
                        self._decode_partial()
        except Exception as exc:
            with self.lock:
                self.error = str(exc)
        finally:
            with self.lock:
                self.active = False
            self._decode(final=True)
            with self.lock:
                self.final = True
                if self.worker is threading.current_thread():
                    self.worker = None

    def _handle_audio_block(self, audio):
        level = float(math.sqrt(float((audio * audio).mean()))) if len(audio) else 0.0
        with self.lock:
            self.level = max(0.0, min(1.0, level * 18))
        speech_like = self._is_speech_like(audio, level)
        if speech_like:
            first_speech = not self._speech_detected
            self._speech_detected = True
            self._silence_start = None
            with self.lock:
                if first_speech and self.pre_roll_chunks:
                    pre_chunks = list(self.pre_roll_chunks)
                    self.audio_chunks.extend([x.copy() for x in pre_chunks])
                    self.segment_chunks.extend([x.copy() for x in pre_chunks])
                    self.pre_roll_chunks = []
                self.audio_chunks.append(audio.copy())
                self.segment_chunks.append(audio.copy())
        else:
            self._update_noise_floor(level)
            if self._speech_detected:
                with self.lock:
                    self.audio_chunks.append(audio.copy())
                    self.segment_chunks.append(audio.copy())
                if self._silence_start is None:
                    self._silence_start = time.time()
                else:
                    silence_ms = (time.time() - self._silence_start) * 1000
                    if silence_ms >= self.short_pause_ms:
                        self._flush_segment_on_pause()
            else:
                with self.lock:
                    self.pre_roll_chunks.append(audio.copy())
                    if len(self.pre_roll_chunks) > self.pre_roll_max_blocks:
                        self.pre_roll_chunks = self.pre_roll_chunks[-self.pre_roll_max_blocks:]

    def _update_noise_floor(self, level):
        return

    def _is_speech_like(self, audio, level):
        return level > self.energy_threshold

    def _segment_audio(self):
        import numpy as np
        with self.lock:
            chunks = list(self.segment_chunks)
        if not chunks:
            return np.zeros(0, dtype="float32")
        return np.concatenate(chunks).astype("float32")

    def _recognize(self, audio):
        with self._recognition_lock:
            stream = self.recognizer.create_stream()
            stream.accept_waveform(self.sample_rate, audio)
            self.recognizer.decode_stream(stream)
            return (stream.result.text or "").strip()

    def _flush_segment_on_pause(self):
        """检测到停顿时，解码当前语音段，并立刻把已校正文本写回输入框。"""
        if not self.recognizer:
            self._silence_start = None
            return
        audio = self._segment_audio()
        if len(audio) < int(self.sample_rate * 0.16):
            with self.lock:
                self.segment_chunks = []
            self._speech_detected = False
            self._silence_start = None
            return
        try:
            raw = self._recognize(audio)
            if raw and self._is_valid_asr_text(raw, audio, partial=True):
                with self.lock:
                    self._pause_text_parts.append(raw)
                    self._pause_audio_parts.append(audio.copy())
                    parts = list(self._pause_text_parts)
                    part_audios = list(self._pause_audio_parts)
                joined = " ".join(parts)
                text = self.correct(joined, final=True, audio=self._audio(), part_audios=part_audios)
                with self.lock:
                    self.raw_text = joined
                    self.committed_text = text
                    self.text = text
                    self.voice_metrics = self.analyze_physical(self._audio(), self.text)
                    self.hint = ""
                    self.last_decode = time.time()
        except Exception as exc:
            with self.lock:
                self.error = str(exc)
                self.active = False
            return
        with self.lock:
            self.segment_chunks = []
        self._speech_detected = False
        self._silence_start = None

    def _flush_segment(self, pause_type=None):
        if not self.recognizer:
            return
        audio = self._segment_audio()
        if len(audio) < int(self.sample_rate * 0.16):
            with self.lock:
                self.segment_chunks = []
            self._speech_detected = False
            self._silence_start = None
            return
        try:
            raw = self._recognize(audio)
            if raw and self._is_valid_asr_text(raw, audio, partial=True):
                text = self.correct(raw, final=False, audio=audio)
                with self.lock:
                    self.raw_text = raw
                    self.committed_text += text
                    self.text = self.committed_text
                    self.voice_metrics = self.analyze_physical(self._audio(), self.text)
                    self.hint = ""
                    self.last_decode = time.time()
        except Exception as exc:
            with self.lock:
                self.error = str(exc)
        finally:
            with self.lock:
                self.segment_chunks = []
            self._speech_detected = False
            self._silence_start = None

    def _decode_partial(self):
        if not self.recognizer:
            return
        audio = self._segment_audio()
        if len(audio) < int(self.sample_rate * 0.16):
            return
        try:
            raw = self._recognize(audio)
            if not raw or not self._is_valid_asr_text(raw, audio, partial=True):
                return
            text = self.correct(raw, final=False, audio=audio)
            with self.lock:
                self.raw_text = raw
                self.text = self.committed_text + text
                self.voice_metrics = self.analyze_physical(self._audio(), self.text)
                self.hint = ""
                self.last_decode = time.time()
        except Exception as exc:
            with self.lock:
                self.error = str(exc)

    def _audio(self):
        import numpy as np
        with self.lock:
            chunks = list(self.audio_chunks)
        if not chunks:
            return np.zeros(0, dtype="float32")
        return np.concatenate(chunks).astype("float32")

    def _decode(self, final=False):
        if final:
            self._decode_final()
            with self.lock:
                self.final = True
        else:
            self._decode_partial()

    def _decode_final(self):
        if not self.recognizer:
            return
        # 1. 刷新最后一段未处理的语音
        audio = self._segment_audio()
        if len(audio) >= int(self.sample_rate * 0.12):
            try:
                raw = self._recognize(audio)
                if raw and self._is_valid_asr_text(raw, audio, partial=True):
                    with self.lock:
                        self._pause_text_parts.append(raw)
                        self._pause_audio_parts.append(audio.copy())
            except Exception as exc:
                with self.lock:
                    self.error = str(exc)
                return

        # 2. 合并所有停顿段，用一个空格分隔
        with self.lock:
            parts = list(self._pause_text_parts)
            part_audios = list(self._pause_audio_parts)
        if not parts:
            return

        raw = " ".join(parts)
        audio = self._audio()

        try:
            if raw and self._is_valid_asr_text(raw, audio):
                text = self.correct(raw, final=True, audio=audio, part_audios=part_audios)
                with self.lock:
                    self.raw_text = raw
                    self.committed_text = text
                    self.text = text
                    self.voice_metrics = self.analyze_physical(audio, self.text)
                    self.hint = ""
                    self.last_decode = time.time()
        except Exception as exc:
            with self.lock:
                self.error = str(exc)

    def _is_valid_asr_text(self, text, audio, partial=False):
        text = re.sub(r"\s+", "", text or "")
        if not text:
            return False
        duration = len(audio) / self.sample_rate if audio is not None else 0
        min_duration = 0.12 if partial else 0.28
        if duration < min_duration:
            return False
        if len(text) <= 1 and duration < (0.12 if partial else 0.7):
            return False
        if re.fullmatch(r"[啊嗯呃额哦诶唉哎呀哈呵哼]+", text) and duration < (0.35 if partial else 0.75):
            return False
        try:
            import numpy as np
            rms = float(np.sqrt(np.mean(audio ** 2))) if audio is not None and len(audio) else 0.0
            if rms < self.energy_threshold * (0.9 if partial else 1.25) and len(text) < (2 if partial else 4):
                return False
        except Exception:
            pass
        return True

    def correct(self, text, final=False, pause_type=None, audio=None, part_audios=None):
        if not text:
            return ""
        for src, dst in self.freq_map.items():
            text = text.replace(src, dst)
        text = self._hotword_correct(text)
        if not final:
            return text
        return self._punctuate(
            text, final=True, pause_type=pause_type, audio=audio,
            part_audios=part_audios,
        )

    def _hotword_correct(self, text):
        """Match Chinese spans without crossing known-word or ASCII boundaries."""
        if not self._pinyin or not self.hotword_pinyin:
            return text
        if not re.search(r"[\u4e00-\u9fff]", text or ""):
            return text
        chars = list(text)
        max_len = min(10, len(chars))
        protected = [False] * len(chars)
        explicit_targets = set(self.freq_map.values())
        for start in range(len(chars)):
            for length in range(min(max_len, len(chars) - start), 1, -1):
                segment = text[start:start + length]
                if segment in self.master_words or segment in explicit_targets:
                    protected[start:start + length] = [True] * length
                    break
        i = 0
        out = []
        while i < len(chars):
            if protected[i]:
                out.append(chars[i])
                i += 1
                continue
            replaced = None
            for n in range(max_len, 1, -1):
                seg = "".join(chars[i:i + n])
                if len(seg) != n:
                    continue
                if not re.fullmatch(r"[\u3400-\u4dbf\u4e00-\u9fff]+", seg):
                    continue
                hw = self.hotword_pinyin.get((n, self._py(seg)))
                if hw and any(
                    protected[i + offset] and char != hw[offset]
                    for offset, char in enumerate(seg)
                ):
                    continue
                if hw and hw != seg:
                    replaced = hw
                    i += n
                    break
            if replaced:
                out.append(replaced)
            else:
                out.append(chars[i])
                i += 1
        return "".join(out)

    def _punctuate(self, text, final=False, pause_type=None, audio=None, part_audios=None):
        if not text:
            return text
        if not final:
            return text
        return self._auto_punctuate_final(text, audio=audio, part_audios=part_audios)

    def _auto_punctuate_final(self, text, audio=None, part_audios=None):
        """Infer punctuation at Chinese pause boundaries; preserve existing text."""
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return text
        if self._is_pure_english(text):
            return text if re.search(r"[.!?]$", text) else (
                text + self._final_end_mark(text, audio=audio)
            )
        return self._fill_punctuation(text, audio=audio, part_audios=part_audios)

    def _normalize_punct(self, text):
        text = re.sub(r"\s+", "", text or "")
        table = str.maketrans({",": "，", ".": "。", "?": "？", "!": "！", ";": "；", ":": "："})
        return text.translate(table)

    def _is_pure_english(self, text):
        return all(c.isascii() for c in text) and any(c.isalpha() for c in text)

    def _fill_punctuation(self, text, audio=None, part_audios=None):
        """在空格处填入标点：空格前疑问语调优先？，完整句填。、连接词前填，，拿不准才逗号。"""
        parts = re.split(
            r"(?<=[\u3400-\u4dbf\u4e00-\u9fff]) +"
            r"(?=[\u3400-\u4dbf\u4e00-\u9fff])",
            text,
        )
        result = []
        part_audios = part_audios or []
        for i, part in enumerate(parts):
            part = part.strip()
            if not part:
                continue
            result.append(part)
            if i < len(parts) - 1:
                acc = "".join(result)
                pause_audio = part_audios[i] if i < len(part_audios) else audio
                punct = self._pause_punctuation(acc, parts[i + 1].strip(), audio=pause_audio)
                result.append(punct)

        full = "".join(result)
        last_audio = part_audios[-1] if part_audios else audio
        if not re.search(r"[。！？.!?]$", full):
            full += self._final_end_mark(full, audio=last_audio)
        return full

    def _pause_punctuation(self, before, after, audio=None):
        """根据前后文判断空格处填什么标点。"""
        # 1. 空格前是疑问语调 + 文本有疑问提示 → 问号；纯陈述不再仅凭音调加问号
        qwords = r"(什么|怎么|怎样|怎么样|为什么|为啥|咋|哪里|哪儿|哪个|哪些|哪位|几|多少|多大|多长|多远|多久|是否|能否|可否|是不是|能不能|可不可以|会不会|对不对|好不好|行不行|要不要|有没有|谁|何时|何地|何人|如何|吗|呢|吧|么)"
        question_like = re.search(qwords, before) or re.search(r"(吗|呢|吧|么)(啊|呀|啦)?$", before)
        if question_like and self._is_question_tone(audio, default=False):
            return "？"
        # 2. 疑问词结尾/近句尾 → 问号
        if re.search(qwords + r"(啊|呀|啦)?$", before):
            return "？" if self._is_question_tone(audio, default=True) else "，"
        if re.search(qwords + r".{0,3}(啊|呀|啦)$", before):
            return "？"
        # 感叹词结尾 → 感叹号
        if re.search(r"(哎呀|哎哟|哎呦|哇|呀|啊|啦|哈|呵|嘿|哼)$", before):
            return "？" if self._is_question_tone(audio) else "！"
        # 2. 后文以连接词开头 → 逗号
        if re.search(r"^(然后|但是|可是|不过|于是|接着|同时|所以|因此|而且|并且|另外|否则|如果|因为|虽然|尽管|只要|除非|既然)", after):
            return "，"
        # 3. 前文看起来是完整句 → 句号
        if self._looks_complete(before):
            return "。"
        # 4. 拿不准 → 逗号
        return "，"

    def _looks_complete(self, text):
        """判断文本是否像完整句（非连接词结尾、非半截话）。"""
        if len(text) < 6:
            return False
        # 不以连接词/介词结尾
        hanging = r"(然后|但是|可是|不过|于是|接着|同时|所以|而且|并且|另外|否则|如果|因为|虽然|尽管|只要|除非|既然|在|从|对|把|被|让|给|跟|和|与|及|或|先|再|就|才|都|还|也|已经|正在|准备|需要|可以|应该|必须|可能|好像)$"
        if re.search(hanging, text):
            return False
        # 不以"我觉得/我想/他说"等开场语结尾
        if re.search(r"(我觉得|我认为|我想|他说|她说|你看|也就是说|换句话说)$", text):
            return False
        return True

    def _final_end_mark(self, text, audio=None):
        """句末标点：疑问语调优先问号，其余句号（感叹词用感叹号）。"""
        en = self._is_pure_english(text)
        if en:
            if re.search(
                r"^(who|what|where|when|why|how|which|whose|whom|is|are|was|were|do|does|did|can|could|will|would|shall|should|have|has|had|may|might)\b",
                text.strip(),
                re.IGNORECASE,
            ):
                return "?"
            return "."
        qwords = r"(什么|怎么|怎样|怎么样|为什么|为啥|咋|哪里|哪儿|哪个|哪些|哪位|几|多少|多大|多长|多远|多久|是否|能否|可否|是不是|能不能|可不可以|会不会|对不对|好不好|行不行|要不要|有没有|谁|何时|何地|何人|如何|吗|呢|吧|么)"
        if re.search(qwords + r"(啊|呀|啦)?$", text):
            return "？"
        if re.search(qwords + r".{0,4}$", text):
            return "？"
        if re.search(qwords + r".{0,3}(啊|呀|啦)$", text):
            return "？"
        if re.search(r"(能|可以|可不可以|会|会不会|要|要不要|有没有|是否|能否|可否).{0,8}(吗|呢|么)$", text):
            return "？"
        question_like = re.search(qwords, text) or re.search(r"(吗|呢|吧|么)(啊|呀|啦)?$", text)
        if question_like and self._is_question_tone(audio, default=False):
            return "？"
        if re.search(r"(哎呀|哎哟|哎呦|唉呀|唉哟|哇|呀|啊|啦|哈|呵|嘿|哼)$", text):
            return "？" if self._is_question_tone(audio) else "！"
        return "。"

    def _is_question_tone(self, audio, default=False):
        if audio is None:
            return default
        try:
            import numpy as np
            x = audio.astype("float32")
            frame = max(1, int(self.sample_rate * 0.08))
            f0 = []
            for i in range(0, len(x) - frame, frame):
                seg = x[i:i + frame]
                rms = float(np.sqrt(np.mean(seg ** 2)))
                if rms < self.energy_threshold * 2.5:
                    continue
                seg = seg - np.mean(seg)
                corr = np.correlate(seg, seg, mode="full")[len(seg) - 1:]
                lo, hi = int(self.sample_rate / 500), int(self.sample_rate / 60)
                if hi >= len(corr):
                    continue
                peak = lo + int(np.argmax(corr[lo:hi]))
                hz = float(self.sample_rate / peak) if peak else 0.0
                if 60 <= hz <= 500:
                    f0.append(hz)
            if len(f0) < 4:
                return default
            head = float(np.median(f0[:max(2, len(f0) // 3)]))
            tail = float(np.median(f0[-max(2, len(f0) // 3):]))
            mean = float(np.mean(f0))
            return (tail - head) > max(6.0, mean * 0.03)
        except Exception:
            return default

    def _metric(self, raw, score):
        return {"raw": round(float(raw), 4), "score": max(1, min(10, int(round(score))))}

    def empty_voice_metrics(self):
        return {
            "avgVolume": self._metric(0.0, 1),
            "peakVolume": self._metric(0.0, 1),
            "speechRate": self._metric(0.0, 1),
            "pauseFrequency": self._metric(0.0, 1),
            "pauseDuration": self._metric(0.0, 1),
            "voiceVariation": self._metric(0.0, 1),
            "endingRise": self._metric(0.0, 1),
            "duration": self._metric(0.0, 1),
        }

    def analyze_physical(self, audio, text=""):
        import numpy as np
        if len(audio) == 0:
            return self.empty_voice_metrics()
        duration = max(0.01, len(audio) / self.sample_rate)
        rms = float(np.sqrt(np.mean(audio ** 2)))
        peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
        speech_rate = len(re.sub(r"\W", "", text or "")) / duration
        frame = max(1, int(self.sample_rate * 0.08))
        energies = [float(np.sqrt(np.mean(audio[i:i + frame] ** 2))) for i in range(0, len(audio) - frame, frame)] or [rms]
        pause_threshold = max(0.010, self.energy_threshold * 0.75)
        pause_frames = sum(1 for x in energies if x < pause_threshold)
        pause_ratio = pause_frames / max(1, len(energies))
        pause_frequency = pause_frames / max(0.1, duration)
        voice_variation = float(np.std(energies)) / max(0.001, rms)
        ending_rise_raw = self._ending_rise(audio)
        return {
            "avgVolume": self._metric(rms, 1 + rms / 0.10 * 9),
            "peakVolume": self._metric(peak, 1 + peak / 0.35 * 9),
            "speechRate": self._metric(speech_rate, 1 + speech_rate / 8.0 * 9),
            "pauseFrequency": self._metric(pause_frequency, 1 + pause_frequency / 5.0 * 9),
            "pauseDuration": self._metric(pause_ratio, 1 + pause_ratio * 9),
            "voiceVariation": self._metric(voice_variation, 1 + voice_variation / 1.5 * 9),
            "endingRise": self._metric(ending_rise_raw, 1 + max(0.0, ending_rise_raw) / 80.0 * 9),
            "duration": self._metric(duration, 1 + min(duration, 10.0) / 10.0 * 9),
        }

    def _ending_rise(self, audio):
        try:
            import numpy as np
            x = audio.astype("float32")
            frame = max(1, int(self.sample_rate * 0.08))
            f0 = []
            for i in range(0, len(x) - frame, frame):
                seg = x[i:i + frame]
                rms = float(np.sqrt(np.mean(seg ** 2)))
                if rms < self.energy_threshold * 2.0:
                    continue
                seg = seg - np.mean(seg)
                corr = np.correlate(seg, seg, mode="full")[len(seg) - 1:]
                lo, hi = int(self.sample_rate / 500), int(self.sample_rate / 60)
                if hi >= len(corr):
                    continue
                peak = lo + int(np.argmax(corr[lo:hi]))
                hz = float(self.sample_rate / peak) if peak else 0.0
                if 60 <= hz <= 500:
                    f0.append(hz)
            if len(f0) < 4:
                return 0.0
            head = float(np.median(f0[:max(2, len(f0) // 3)]))
            tail = float(np.median(f0[-max(2, len(f0) // 3):]))
            return tail - head
        except Exception:
            return 0.0

    def mood_hint(self, mood):
        return ""

    def snapshot(self):
        with self.lock:
            remaining = 0
            if self.active and self.started_at:
                remaining = max(0, self.max_seconds - (time.time() - self.started_at))
            return {
                "initialized": self.initialized,
                "loading": self.loading,
                "active": self.active,
                "final": self.final,
                "remaining": round(remaining, 1),
                "level": self.level,
                "rawText": self.raw_text,
                "text": self.text,
                "displayText": sanitize_display_text(self.text),
                "voiceMetrics": self.voice_metrics,
                "thresholdLevel": self.threshold_level,
                "energyThreshold": self.energy_threshold,
                "error": self.error,
                "hotwordCount": len(self.hotwords),
                "masterReady": self.master_ready,
                "masterCount": len(self.master_words),
                "inputDevice": self.input_device_name,
            }
