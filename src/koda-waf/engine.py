"""Dependency-free request inspection engine for Koda-WAF."""

from __future__ import annotations

import base64
import binascii
import html
import math
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote


@dataclass(frozen=True)
class RequestView:
    method: str = "GET"
    path: str = "/"
    query: str = ""
    headers: str = ""
    body: str = ""

    def text(self) -> str:
        return f"method={self.method} path={self.path} query={self.query} headers={self.headers} body={self.body}"


@dataclass(frozen=True)
class Decision:
    blocked: bool
    score: float
    reasons: tuple[str, ...]


SIGNATURES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("sql_injection", re.compile(
        r"(?:\bunion\s+(?:all\s+)?select\b|\bselect\s*\([^)]*\)\s*from\b|"
        r"\b(?:sleep|benchmark|pg_sleep|json_extract|json_depth)\s*\(|"
        r"\b(?:or|and)\s+[-'\"\w]+\s*(?:=|like|!=|<>)\s*[-'\"\w]+|"
        r"\bdeclare\s+@\w+|\bxp_cmdshell\b|\binformation_schema\b)", re.I)),
    ("xss", re.compile(
        r"(?:<\s*/?\s*(?:script|iframe|svg|body|img|object|embed|meta|base|link)\b|"
        r"\bon[a-z]{3,}\s*=|javascript\s*:|__proto__|\.constructor\s*\(|"
        r"(?:alert|prompt|confirm|fetch)\s*(?:\.|\(|`)|setinterval\s*\()", re.I)),
    ("path_traversal", re.compile(
        r"(?:\.\.[/\\]|/etc/(?:passwd|shadow)|\\windows\\|c\$\\users|ntuser\.dat)", re.I)),
    ("command_injection", re.compile(
        r"(?:(?:;|\|\||&&|\||`|\$\(|--exec=)\s*(?:\$ifs\$\d*)*"
        r"(?:cat(?:\s|$)|id(?:\s|$)|whoami\b|wget\s+|curl\s+|bash(?:\s|$)|sh(?:\s|$)|"
        r"nc\s+|ncat\s+|powershell\b|cmd(?:\.exe)?\b|ls(?:\s|$)|dir(?:\s|$)|getent\s+|ping\s+)|"
        r"\(\)\s*\{\s*:\s*;\s*\}|/bin/(?:ba)?sh\b)", re.I | re.S)),
    ("ssrf", re.compile(
        r"(?:169\.254\.169\.254|metadata\.google\.internal|(?:localhost|127\.0\.0\.1)(?::\d+)?|"
        r"0x7f000001|2130706433|gopher://|dict://|file:///|burpcollaborator\.net|interact\.sh)", re.I)),
    ("crlf_or_mail_injection", re.compile(
        r"(?:(?:\r|\n|%0[ad]).{0,60}(?:set-cookie\s*:|rcpt\s+to\s*:|capability|fetch\s+\d+|quit))",
        re.I | re.S)),
    ("ldap_injection", re.compile(
        r"(?:\(\s*[&|!]\s*\([^)]*(?:uid|objectclass|userpassword)|objectclass\s*=\s*\*|"
        r"userpassword:2\.5\.13\.18)", re.I)),
    ("nosql_injection", re.compile(
        r"(?:\$(?:where|or|ne|gt)\b|db\.\w+\.(?:insert|find)\s*\(|"
        r"new\s+date\s*\(\).{0,80}while\s*\()", re.I | re.S)),
    ("server_side_include", re.compile(r"<!--\s*#\s*(?:exec|include|echo|config)\b", re.I)),
    ("template_injection", re.compile(
        r"(?:\{\{\s*(?:\d+\s*[*+/\-]\s*\d+|(?:config|self|request|cycler|joiner|namespace)\b)[^}]*\}\}|"
        r"#\{\s*\d+\s*[*+/\-]\s*\d+\s*\}|<#assign\b|\?new\s*\(\)|"
        r"\$\{\s*(?:\d+\s*[*+/\-]\s*\d+|jndi:|[a-z_]\w*\s*\()[^}]*\})", re.I)),
    ("xml_external_entity", re.compile(
        r"(?:<!doctype\b.{0,300}<!entity\b|<!entity\b[^>]*(?:system|public)|"
        r"<xs:include\b|xsi:schemalocation\s*=|expect://)", re.I | re.S)),
)

BASE64_TOKEN = re.compile(r"(?<![A-Za-z0-9+/])([A-Za-z0-9+/]{12,}={0,2})(?![A-Za-z0-9+/])")
TOKEN = re.compile(r"[a-z0-9_$./:@-]{2,}|[^\w\s]{2,}", re.I)


def decoded_variants(value: str, max_variants: int = 32) -> list[str]:
    """Decode common WAF evasions with strict work limits."""
    queue = [value]
    seen: set[str] = set()
    output: list[str] = []
    while queue and len(output) < max_variants:
        current = queue.pop(0)
        if current in seen:
            continue
        seen.add(current)
        normalized = current.replace("\x00", "").replace("\\u0027", "'").replace("\\u0022", '"')
        output.append(normalized.lower())
        for candidate in (unquote(normalized), html.unescape(normalized)):
            if candidate != normalized and candidate not in seen:
                queue.append(candidate)

        for match in BASE64_TOKEN.finditer(normalized):
            token = match.group(1)
            if len(token) > 1_000_000:
                continue
            try:
                padded = token + "=" * (-len(token) % 4)
                decoded = base64.b64decode(padded, validate=True).decode("utf-8", errors="ignore")
            except (binascii.Error, ValueError):
                continue
            printable = sum(character.isprintable() or character.isspace() for character in decoded)
            if decoded and printable / len(decoded) >= 0.85 and decoded not in seen:
                queue.append(decoded)
    return output


def extract_features(request: RequestView) -> set[str]:
    variants = decoded_variants(request.text())
    features = {f"method:{request.method.lower()}"}
    request_content = " ".join((request.path, request.query, request.headers, request.body))
    content_words = [token for token in TOKEN.findall(request_content) if token[:1].isalnum()]
    context = "prose" if len(content_words) >= 12 else "compact"

    for variant in variants:
        tokens = TOKEN.findall(variant)
        features.update(f"token:{token}" for token in tokens)
        features.update(f"bigram:{left}|{right}" for left, right in zip(tokens, tokens[1:]))
        words = [token for token in tokens if token[:1].isalnum()]
        features.add(f"word-count:{min(len(words) // 4, 20)}")
        features.add(
            f"alpha-ratio:{min(int(sum(char.isalpha() for char in variant) / max(len(variant), 1) * 10), 10)}"
        )
        for name, signature in SIGNATURES:
            if signature.search(variant):
                features.add(f"signature:{name}")
                features.add(f"signature-context:{name}:{context}")
    features.add(f"path-depth:{min(request.path.count('/'), 8)}")
    features.add(f"body-size:{min(len(request.body) // 64, 16)}")
    features.add(f"query-size:{min(len(request.query) // 32, 16)}")
    return features


class WAFModel:
    format_version = 1

    def __init__(self, weights: dict[str, float], bias: float, threshold: float = 0.5) -> None:
        self.weights = weights
        self.bias = bias
        self.threshold = threshold

    def probability(self, request: RequestView) -> float:
        features = extract_features(request)
        evidence = sum(self.weights.get(feature, 0.0) for feature in features)
        score = self.bias + evidence / math.sqrt(max(len(features), 1))
        score = max(min(score, 60.0), -60.0)
        return 1.0 / (1.0 + math.exp(-score))

    def save(self, path: Path) -> None:
        payload = {
            "format": "koda-waf",
            "version": self.format_version,
            "weights": self.weights,
            "bias": self.bias,
            "threshold": self.threshold,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as file:
            pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> "WAFModel":
        with path.open("rb") as file:
            payload = pickle.load(file)
        if payload.get("format") != "koda-waf" or payload.get("version") != cls.format_version:
            raise ValueError("Unsupported Koda-WAF model format")
        return cls(
            {str(key): float(value) for key, value in payload["weights"].items()},
            float(payload["bias"]),
            float(payload["threshold"]),
        )


class KodaWAF:
    def __init__(self, model: WAFModel) -> None:
        self.model = model

    @classmethod
    def from_model(cls, path: Path) -> "KodaWAF":
        return cls(WAFModel.load(path))

    def inspect(self, request: RequestView) -> Decision:
        variants = decoded_variants(request.text())
        reasons: list[str] = []
        for name, signature in SIGNATURES:
            if any(signature.search(variant) for variant in variants):
                reasons.append(name)
        score = self.model.probability(request)
        blocked = score >= self.model.threshold
        if blocked and not reasons:
            reasons.append("learned_pattern")
        return Decision(blocked, score, tuple(reasons))
