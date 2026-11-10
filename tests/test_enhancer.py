import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

from thinkingears_asr import ASREnhancer


class EnhancerTests(unittest.TestCase):
    def enhancer(self, **kwargs):
        return ASREnhancer(use_builtin_lexicon=False, **kwargs)

    def test_corrects_homophone_to_a_selected_hotword(self):
        enhancer = self.enhancer(hotwords=["共振峰"])
        self.assertEqual(enhancer.enhance("测量供振峰"), "测量共振峰")

    def test_known_word_is_preserved_instead_of_replaced_by_a_hotword(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "words.json"
            path.write_text(json.dumps({"继承": {}}), encoding="utf-8")
            enhancer = self.enhancer(hotwords=["集成"], lexicon_path=path)
            self.assertEqual(enhancer.enhance("继承"), "继承")

    def test_protected_word_cannot_be_rewritten_from_its_second_character(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "words.txt"
            path.write_text("模型\n", encoding="utf-8")
            enhancer = self.enhancer(hotwords=["型章"], lexicon_path=path)
            self.assertEqual(enhancer.enhance("模型张"), "模型张")

    def test_ambiguous_hotwords_do_not_choose_an_arbitrary_candidate(self):
        enhancer = self.enhancer(hotwords=["集成", "继承"])
        self.assertEqual(enhancer.enhance("寄呈"), "寄呈")

    def test_hotwords_can_be_replaced_and_cleared(self):
        enhancer = self.enhancer(hotwords=["共振峰"])
        enhancer.set_hotwords([])
        self.assertEqual(enhancer.enhance("供振峰"), "供振峰")
        enhancer.set_hotwords(["共振峰"])
        self.assertEqual(enhancer.enhance("供振峰"), "共振峰")

    def test_no_game_replacement_rules_are_enabled_by_default(self):
        self.assertEqual(self.enhancer().enhance("人无"), "人无")

    def test_explicit_corrections_are_optional(self):
        enhancer = self.enhancer(corrections={"人无": "任务"})
        self.assertEqual(enhancer.enhance("人无"), "任务")

    def test_english_spacing_and_existing_punctuation_are_preserved(self):
        text = "hello world, AI speech recognition."
        self.assertEqual(self.enhancer().enhance(text), text)

    def test_empty_and_mixed_language_inputs(self):
        enhancer = self.enhancer()
        self.assertEqual(enhancer.enhance(""), "")
        self.assertEqual(enhancer.enhance("调用 ASR API"), "调用 ASR API")

    def test_final_english_punctuation_preserves_word_boundaries(self):
        self.assertEqual(
            self.enhancer().enhance("hello world", final=True), "hello world."
        )

    def test_missing_custom_lexicon_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                self.enhancer(lexicon_path=Path(directory) / "missing.json")

    def test_invalid_custom_lexicon_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text("{broken", encoding="utf-8")
            with self.assertRaises(ValueError):
                self.enhancer(lexicon_path=path)

    def test_bundled_lexicon_protects_a_real_chinese_word(self):
        enhancer = ASREnhancer(hotwords=["集成"])
        self.assertEqual(enhancer.lexicon_count, 578895)
        self.assertEqual(enhancer.enhance("继承"), "继承")

    def test_text_only_use_does_not_import_the_recognition_backend(self):
        code = (
            "import sys; from thinkingears_asr import ASREnhancer; "
            "ASREnhancer(use_builtin_lexicon=False).enhance('hello'); "
            "assert 'sherpa_onnx' not in sys.modules; "
            "assert 'sounddevice' not in sys.modules"
        )
        subprocess.run([sys.executable, "-c", code], check=True, capture_output=True)

    def test_metrics_measure_silence_without_nonfinite_values(self):
        metrics = self.enhancer().analyze(np.zeros(16000, dtype=np.float32), "你好")
        self.assertEqual(metrics["avgVolume"]["raw"], 0.0)
        self.assertEqual(metrics["duration"]["raw"], 1.0)
        self.assertEqual(metrics["speechRate"]["raw"], 2.0)
        self.assertTrue(all(math.isfinite(item["raw"]) for item in metrics.values()))

    def test_cli_enhances_text_without_a_model(self):
        result = subprocess.run(
            [sys.executable, "-m", "thinkingears_asr", "enhance", "hello world"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            env={**os.environ, "PYTHONUTF8": "1"},
        )
        self.assertEqual(result.stdout.strip(), "hello world")

    def test_model_failure_is_visible_to_the_caller(self):
        with tempfile.TemporaryDirectory() as directory:
            enhancer = self.enhancer(model_dir=directory)
            with self.assertRaises(RuntimeError):
                enhancer.initialize()
            self.assertFalse(enhancer.snapshot()["initialized"])
            self.assertTrue(enhancer.snapshot()["error"])

    def test_silence_does_not_trigger_a_model_or_hallucinated_text(self):
        enhancer = self.enhancer()
        for audio in ([], np.zeros(16000, dtype=np.float32)):
            result = enhancer.transcribe(audio)
            self.assertEqual(result["text"], "")
            self.assertEqual(result["raw_text"], "")
            self.assertFalse(enhancer.initialized)

    def test_invalid_audio_is_rejected_before_model_initialization(self):
        enhancer = self.enhancer()
        for audio in (np.zeros((2, 16000)), [float("nan")], [float("inf")]):
            with self.assertRaises(ValueError):
                enhancer.transcribe(audio)
            self.assertFalse(enhancer.initialized)

    def test_hotword_replacement_cannot_consume_ascii_text(self):
        enhancer = self.enhancer(hotwords=["共振峰"])
        self.assertEqual(
            enhancer.enhance("供振峰, hello"), "共振峰, hello"
        )

    def test_longer_hotword_cannot_override_a_known_word_prefix(self):
        enhancer = ASREnhancer(hotwords=["模形章"])
        self.assertEqual(enhancer.enhance("模型张"), "模型张")

    def test_final_punctuation_preserves_numbers_versions_and_domains(self):
        enhancer = self.enhancer()
        for text, expected in (
            ("价格3.14元", "价格3.14元。"),
            ("访问example.com", "访问example.com。"),
            ("使用v1.2.3版本", "使用v1.2.3版本。"),
        ):
            with self.subTest(text=text):
                self.assertEqual(enhancer.enhance(text, final=True), expected)

    def test_final_punctuation_preserves_existing_sentence_terminators(self):
        enhancer = self.enhancer()
        for text in ("hello world?", "hello!", "你好！", "这是结尾。"):
            with self.subTest(text=text):
                self.assertEqual(enhancer.enhance(text, final=True), text)

    def test_final_punctuation_preserves_spaces_within_mixed_language(self):
        self.assertEqual(
            self.enhancer().enhance("调用 ASR API", final=True), "调用 ASR API。"
        )

    def test_english_question_words_require_word_boundaries(self):
        enhancer = self.enhancer()
        for text in ("Whatever works", "Wholesale prices"):
            self.assertEqual(enhancer.enhance(text, final=True), text + ".")
        self.assertEqual(
            enhancer.enhance("What is this", final=True), "What is this?"
        )

    def test_complex_audio_is_rejected_instead_of_silently_cast(self):
        enhancer = self.enhancer()
        audio = np.array([complex(0.1, np.nan)])
        for operation in (enhancer.analyze, enhancer.transcribe):
            with self.assertRaises(ValueError):
                operation(audio)

    def test_recording_cannot_restart_while_old_worker_is_finishing(self):
        enhancer = self.enhancer()
        enhancer.initialized = True
        release = threading.Event()
        worker = threading.Thread(target=lambda: release.wait(5))
        worker.start()
        enhancer.worker = worker
        try:
            with patch.object(enhancer, "_record_loop"):
                ok, _ = enhancer.start()
            self.assertFalse(ok)
            self.assertIs(enhancer.worker, worker)
        finally:
            release.set()
            worker.join()
            if enhancer.worker is not None and enhancer.worker is not worker:
                enhancer.worker.join()

    def test_new_recording_clears_previous_device_error(self):
        enhancer = self.enhancer()
        enhancer.initialized = True
        enhancer.error = "previous input device failed"
        release = threading.Event()
        try:
            with patch.object(enhancer, "_record_loop", lambda: release.wait(5)):
                ok, _ = enhancer.start()
            self.assertTrue(ok)
            self.assertEqual(enhancer.snapshot()["error"], "")
        finally:
            release.set()
            enhancer.stop()

    def test_failed_final_decode_reports_failure_and_keeps_audio(self):
        class FailedDecoder:
            def create_stream(self):
                raise RuntimeError("decoder failed")

        enhancer = self.enhancer()
        enhancer.recognizer = FailedDecoder()
        enhancer.segment_chunks = [np.full(8000, 0.1, dtype=np.float32)]
        enhancer._decode_final()
        self.assertIn("decoder failed", enhancer.snapshot()["error"])
        self.assertEqual(len(enhancer.segment_chunks), 1)

    def test_failed_pause_decode_reports_failure_and_keeps_audio(self):
        class FailedDecoder:
            def create_stream(self):
                raise RuntimeError("decoder failed")

        enhancer = self.enhancer()
        enhancer.recognizer = FailedDecoder()
        enhancer.segment_chunks = [np.full(8000, 0.1, dtype=np.float32)]
        enhancer._flush_segment_on_pause()
        self.assertIn("decoder failed", enhancer.snapshot()["error"])
        self.assertEqual(len(enhancer.segment_chunks), 1)

    def test_sync_and_async_initialization_share_one_backend_load(self):
        enhancer = self.enhancer()
        entered = threading.Event()
        release = threading.Event()
        errors = []
        loads = []

        def load_backend():
            loads.append(threading.get_ident())
            entered.set()
            if not release.wait(3):
                raise RuntimeError("test initializer timed out")
            with enhancer.lock:
                enhancer.initialized = True
                enhancer.loading = False

        def initialize_again():
            try:
                enhancer.initialize()
            except Exception as exc:
                errors.append(exc)

        with patch.object(enhancer, "_initialize_backend", load_backend):
            enhancer.init_async()
            self.assertTrue(entered.wait(2))
            synchronous = threading.Thread(target=initialize_again)
            synchronous.start()
            synchronous.join(timeout=0.05)
            self.assertTrue(synchronous.is_alive())
            release.set()
            synchronous.join(timeout=2)
        self.assertFalse(synchronous.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(len(loads), 1)
        self.assertTrue(enhancer.initialized)
        self.assertFalse(enhancer.loading)


if __name__ == "__main__":
    unittest.main()
