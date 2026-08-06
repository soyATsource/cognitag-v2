"""
Facet 前提の辞書スキーマと永続化層。

旧 meta_dictionary.json（単語 -> タグ配列 のフラット構造）を置き換える。
主な変更点:
  - 各エントリが facets（4軸の重み）を第一級のデータとして持つ
  - provenance（出自・試行回数・分散・検証済みフラグ）を必須にする
  - facet_axes をファイル冒頭に持ち、軸定義を変えても既存データの解釈が壊れない
  - 保存は os.replace によるアトミック書き込み（中断しても壊れない）
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .facets import FACET_KEYS

SCHEMA_VERSION = "2.0"

# タグとして明らかに壊れているものを弾くための検出パターン
MOJIBAKE_CHARS = set("縺繧蜊隧譁鬟髮莠螳邨蜃驛逅譛盤")


@dataclass
class Provenance:
    """このエントリがどこから来て、どれだけ信頼できるか"""

    source: str = "llm_annotation"  # llm_annotation | manual | imported
    model: str = ""
    trials: int = 0
    # 破棄した先頭試行の回数。Ollama のプロンプトキャッシュ未温存による
    # 1回目の不安定さを分散から排除するため（facet_annotator.py 参照）。
    # 旧方式で注釈した既存エントリは 0 のままなので、新旧を後から切り分けられる。
    warmup_trials: int = 0
    variance: float = 0.0  # 全軸の分散の最大値
    verified: bool = False
    frequency: int = 0  # コーパス中の出現回数
    categories: list[str] = field(default_factory=list)  # 出現したカテゴリ
    annotated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Entry:
    """辞書の1エントリ"""

    word: str
    pos: str = ""
    facets: dict[str, float] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    provenance: Provenance = field(default_factory=Provenance)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pos": self.pos,
            "facets": self.facets,
            "tags": self.tags,
            "provenance": self.provenance.to_dict(),
        }

    @classmethod
    def from_dict(cls, word: str, data: dict[str, Any]) -> Entry:
        prov_data = data.get("provenance", {}) or {}
        prov = Provenance(
            source=prov_data.get("source", "imported"),
            model=prov_data.get("model", ""),
            trials=int(prov_data.get("trials", 0) or 0),
            warmup_trials=int(prov_data.get("warmup_trials", 0) or 0),
            variance=float(prov_data.get("variance", 0.0) or 0.0),
            verified=bool(prov_data.get("verified", False)),
            frequency=int(prov_data.get("frequency", 0) or 0),
            categories=list(prov_data.get("categories", []) or []),
            annotated_at=prov_data.get("annotated_at", ""),
        )
        return cls(
            word=word,
            pos=data.get("pos", ""),
            facets=dict(data.get("facets", {}) or {}),
            tags=list(data.get("tags", []) or []),
            provenance=prov,
        )


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def is_valid_word(word: str) -> tuple[bool, str]:
    """辞書のキーとして妥当な語かを検証する。

    旧辞書で発生していた破損パターンを構造的に弾く:
      - キーが '#' で始まる（タグが単語欄に混入）
      - 文字化け文字を含む
      - 空文字・過度に長い
      - 記号のみ
    """
    if not word or not word.strip():
        return False, "empty"
    word = word.strip()
    if word.startswith("#"):
        return False, "tag_as_key"
    if len(word) > 30:
        return False, "too_long"
    if any(ch in MOJIBAKE_CHARS for ch in word):
        return False, "mojibake"
    if not any(ch.isalnum() for ch in word):
        return False, "symbols_only"
    return True, ""


def clean_tags(raw: object) -> list[str]:
    """タグ配列を正規化する。

    旧実装では読点(、)での分割漏れ、空タグ '#'、空白混入 '# #foo' が
    そのまま登録されていた。ここで全て吸収する。
    """
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []

    result: list[str] = []
    for item in raw:
        text = str(item)
        # 読点・カンマ・スラッシュいずれでも分割する（旧実装はカンマのみ）
        for part in text.replace("、", ",").replace("/", ",").split(","):
            tag = part.strip().strip('"\'[]「」【】*')
            tag = tag.replace(" ", "").replace("\u3000", "")
            # '# #foo' -> '##foo' となるため、先頭の # を1つに畳む
            tag = re.sub(r"^#+", "#", tag)
            if not tag or tag == "#":
                continue
            if any(ch in MOJIBAKE_CHARS for ch in tag):
                continue
            if not tag.startswith("#"):
                tag = "#" + tag
            if len(tag) > 30:
                continue
            if tag not in result:
                result.append(tag)
    return result


class DictionaryV2:
    """Facet 付き辞書の読み書き"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.entries: dict[str, Entry] = {}
        # 最後に読み込んだファイルの updated_at。API の stats 応答で返すため、
        # 4MB超の JSON を再パースせずに済むよう load 時に控えておく。
        self.updated_at: str = ""

    def load(self) -> DictionaryV2:
        if not self.path.exists():
            self.entries = {}
            self.updated_at = ""
            return self
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Dictionary root must be an object: {self.path}")
        self.updated_at = str(data.get("updated_at", "") or "")
        raw_entries = data.get("entries", {})
        self.entries = {
            word: Entry.from_dict(word, payload)
            for word, payload in raw_entries.items()
            if isinstance(payload, dict)
        }
        return self

    def save(self) -> None:
        """アトミック保存。中断しても既存ファイルは壊れない。"""
        payload = {
            "schema_version": SCHEMA_VERSION,
            "facet_axes": list(FACET_KEYS),
            "updated_at": now_iso(),
            "entry_count": len(self.entries),
            "entries": {
                word: entry.to_dict()
                for word, entry in sorted(self.entries.items())
            },
        }
        text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=str(self.path.parent), delete=False
        ) as tmp:
            tmp.write(text)
            tmp.write("\n")
            tmp_path = tmp.name
        os.replace(tmp_path, self.path)

    def add(self, entry: Entry) -> bool:
        ok, _reason = is_valid_word(entry.word)
        if not ok:
            return False
        self.entries[entry.word] = entry
        return True

    def has(self, word: str) -> bool:
        return word in self.entries

    def verified_entries(self) -> dict[str, Entry]:
        return {w: e for w, e in self.entries.items() if e.provenance.verified}

    def stats(self) -> dict[str, Any]:
        total = len(self.entries)
        verified = len(self.verified_entries())
        axis_means: dict[str, float] = {}
        for key in FACET_KEYS:
            values = [
                e.facets.get(key, 0.0)
                for e in self.entries.values()
                if e.facets
            ]
            axis_means[key] = round(sum(values) / len(values), 4) if values else 0.0
        return {
            "total": total,
            "verified": verified,
            "unverified": total - verified,
            "axis_means": axis_means,
        }
