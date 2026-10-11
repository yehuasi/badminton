# -*- coding: utf-8 -*-
"""核心逻辑单元测试（unittest，无需额外依赖）。"""

import os
import sys
import unittest

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from app.analyzer import ShotAnalyzer, _hit_group, detect_hit_frame  # noqa: E402
from app.classes import SHOT_CLASSES, group_allowed_classes, shot_group  # noqa: E402
from app.features import F, add_kinematics, compute_features, feature_dict  # noqa: E402
from app.fusion import fuse_views  # noqa: E402
from app.score import score_standardness  # noqa: E402
from app.synth import generate_sequence  # noqa: E402


class TestClasses(unittest.TestCase):
    def test_grouping(self):
        self.assertEqual(shot_group("高远球"), "overhead")
        self.assertEqual(shot_group("挑球"), "non_overhead")
        self.assertEqual(set(group_allowed_classes("overhead")), {"高远球", "杀球", "吊球"})


class TestFeatures(unittest.TestCase):
    def test_add_kinematics_shape(self):
        seq = np.random.rand(24, F).astype(np.float32)
        out = add_kinematics(seq)
        self.assertEqual(out.shape, (24, F + 8))

    def test_compute_features_shape(self):
        kpts = np.random.rand(17, 3).astype(np.float32)
        kpts[:, 2] = 1.0
        feat = compute_features(kpts, handedness="right")
        self.assertEqual(feat.shape, (F,))
        self.assertEqual(len(feature_dict(kpts)), F)


class TestGate(unittest.TestCase):
    def test_group_gate_on_synthetic(self):
        expect = {0: "non_overhead", 1: "non_overhead", 2: "overhead",
                  3: "overhead", 4: "overhead", 5: "non_overhead", 6: "non_overhead"}
        for cid in range(7):
            seq = generate_sequence(cid, T=24, rng=np.random.default_rng(cid))
            hi = detect_hit_frame(seq)
            self.assertEqual(_hit_group(seq, hi), expect[cid], SHOT_CLASSES[cid])


class TestScore(unittest.TestCase):
    def test_in_range_full_score(self):
        score, errors = score_standardness(
            {"arm_raise": 160.0, "elbow": 165.0, "wrist_above_head": 0.5}, action="高远球"
        )
        self.assertGreaterEqual(score, 90.0)
        self.assertEqual(errors, [])

    def test_out_of_range_penalty(self):
        score, errors = score_standardness(
            {"arm_raise": 80.0, "elbow": 90.0, "wrist_above_head": -1.0}, action="高远球"
        )
        self.assertLess(score, 70.0)
        self.assertTrue(len(errors) > 0)


class TestFusion(unittest.TestCase):
    def test_fusion_total(self):
        front = {"shot_type": "高远球", "confidence": 0.6, "standardness": 80.0,
                 "coherence": 70.0, "footwork": 60.0, "errors": [], "advice": []}
        side = {"shot_type": "高远球", "confidence": 0.7, "standardness": 76.0,
                "coherence": 74.0, "footwork": 55.0, "errors": [], "advice": []}
        m = fuse_views(front, side)
        self.assertEqual(m["shot_type"], "高远球")
        self.assertAlmostEqual(m["standardness"], 77.8, places=1)
        self.assertGreater(m["confidence"], 0.7)
        self.assertIsNotNone(m["total"])


class TestAnalyzer(unittest.TestCase):
    def test_analyzer_runs_without_classifier(self):
        analyzer = ShotAnalyzer(classifier=None, window=24)
        seq = generate_sequence(2, T=24)
        res = analyzer.analyze_sequence(seq)
        self.assertIn("total", res)
        self.assertIn("shot_type", res)


class TestMotionGate(unittest.TestCase):
    def _feed(self, analyzer, seq):
        out = []
        for t in range(seq.shape[0]):
            res = analyzer.update_feature(seq[t])
            if res is not None:
                out.append(res)
        return out

    def test_stroke_has_result(self):
        from app.classifier import ShotClassifierRunner

        classifier = ShotClassifierRunner("models/shot_classifier.pt", T=24)
        a = ShotAnalyzer(classifier=classifier, window=24)
        s1 = generate_sequence(2, T=24, rng=np.random.default_rng(0))
        idle = np.tile(s1[0], (16, 1))
        out = self._feed(a, np.concatenate([s1, idle], axis=0))
        self.assertGreaterEqual(len(out), 1)

    def test_idle_no_result(self):
        from app.classifier import ShotClassifierRunner

        classifier = ShotClassifierRunner("models/shot_classifier.pt", T=24)
        a = ShotAnalyzer(classifier=classifier, window=24)
        s1 = generate_sequence(2, T=24, rng=np.random.default_rng(0))
        out = self._feed(a, np.tile(s1[0], (40, 1)))
        self.assertEqual(len(out), 0)


if __name__ == "__main__":
    unittest.main()
