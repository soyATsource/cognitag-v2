"""
フェーズB: 候補語の抽出と頻度フィルタ

コーパス（フェーズA の出力）を形態素解析し、実際に出現した語だけを
候補として取り出す。出現1回だけの語は原則として捨てる。

【なぜ頻度フィルタが決定的に重要か】
旧 TagGraph のスコア式は score = weight / sqrt(freq_a * freq_b) であり、
頻度1のタグ同士が1回共起すると 1/sqrt(1*1) = 1.0 という最大スコアになる。
つまりハルシネーション由来の孤立語が、本当に意味のある頻出語の関連度を
上回ってしまう。頻度2以上に絞ることで、この破綻を入口で防ぐ。
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .stopwords import EXCLUDED_POS, EXCLUDED_SUBPOS, is_stopword

# 内容語として採用する品詞
# ※接続詞は logical 軸のアンカー語彙として価値が高いため残す
CONTENT_POS = {"名詞", "動詞", "形容詞", "副詞", "接続詞", "連体詞"}

# 対話の話者ラベル等、コーパス特有のノイズ
SPEAKER_PATTERN = re.compile(r"^[ABＡＢ]\s*[:：]\s*", re.MULTILINE)

# LLM が出力する Markdown 強調記法（**用語**）
MARKDOWN_PATTERN = re.compile(r"[*_`#]+")

# ひらがな1文字だけの語などを弾く
MIN_LENGTH = 2


@dataclass
class Candidate:
    """注釈待ちの候補語"""

    word: str
    pos: str
    frequency: int = 0
    categories: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "pos": self.pos,
            "frequency": self.frequency,
            "categories": self.categories,
        }


def is_content_word(token: dict[str, str]) -> bool:
    """内容語（Facet 重みを持つ意味のある語）かを判定する。

    候補語抽出（CandidateExtractor）と Facet 算出（FacetScorer）の
    両方から呼ばれる共通判定。辞書構築時と参照時で判定基準がずれると、
    「辞書にあるはずの語が引けない」という不整合が起きるため、
    必ずこの関数を単一の判定基準として使うこと。
    """
    pos = token["pos"]
    if pos in EXCLUDED_POS:
        return False
    if pos not in CONTENT_POS and pos != "未知":
        return False
    if token.get("subpos") in EXCLUDED_SUBPOS:
        return False

    base = token["base"]
    if len(base) < MIN_LENGTH:
        return False
    if base.isdigit():
        return False
    if not any(ch.isalnum() for ch in base):
        return False
    # 機能語（こと・もの・使う 等）は Facet 重みを持つ意味がない
    if is_stopword(base):
        return False
    # ひらがなのみ2文字の語は機能語である可能性が高い
    if len(base) == 2 and all("ぁ" <= ch <= "ん" for ch in base):
        return False
    return True


class Tokenizer:
    """SudachiPy を優先し、無い場合は正規表現でフォールバックする"""

    def __init__(self) -> None:
        self._sudachi = None
        self._split_mode = None
        try:
            from sudachipy import Dictionary, SplitMode

            self._sudachi = Dictionary().create()
            self._split_mode = SplitMode.C
        except Exception:
            self._sudachi = None

    @property
    def available(self) -> bool:
        return self._sudachi is not None

    # Sudachi の入力上限は 49,149 バイト。余裕を持たせて分割する。
    MAX_CHUNK_BYTES = 40000

    def _split_chunks(self, text: str) -> list[str]:
        """Sudachi の入力上限を超えないよう、改行・句点を境に分割する"""
        if len(text.encode("utf-8")) <= self.MAX_CHUNK_BYTES:
            return [text]

        chunks: list[str] = []
        buffer = ""
        buffer_bytes = 0
        # 改行を優先し、それでも長い場合は句点で切る
        units = text.replace("。", "。\n").split("\n")
        for unit in units:
            unit_bytes = len(unit.encode("utf-8")) + 1
            if unit_bytes > self.MAX_CHUNK_BYTES:
                # 単一行が上限を超える異常ケースは文字数で強制分割
                if buffer:
                    chunks.append(buffer)
                    buffer, buffer_bytes = "", 0
                step = self.MAX_CHUNK_BYTES // 4  # UTF-8日本語は最大4バイト
                for i in range(0, len(unit), step):
                    chunks.append(unit[i:i + step])
                continue
            if buffer_bytes + unit_bytes > self.MAX_CHUNK_BYTES:
                chunks.append(buffer)
                buffer, buffer_bytes = "", 0
            buffer += unit + "\n"
            buffer_bytes += unit_bytes
        if buffer:
            chunks.append(buffer)
        return chunks

    def analyze(self, text: str) -> list[dict[str, str]]:
        if self._sudachi is None:
            # フォールバック: 品詞が取れないため精度は大きく落ちる
            surfaces = re.findall(
                r"[一-龥々〆ヵヶ]+|[ァ-ヴー]+|[A-Za-z][A-Za-z0-9_+#.-]*", text
            )
            return [
                {"surface": s, "base": s, "pos": "未知", "subpos": "*"}
                for s in surfaces
            ]

        tokens = []
        for chunk in self._split_chunks(text):
            for morpheme in self._sudachi.tokenize(chunk, self._split_mode):
                pos = morpheme.part_of_speech()
                tokens.append(
                    {
                        "surface": morpheme.surface(),
                        "base": morpheme.dictionary_form(),
                        "pos": pos[0],
                        "subpos": pos[1],
                    }
                )
        return tokens


class CandidateExtractor:
    """コーパス群から候補語を抽出する"""

    def __init__(self, min_frequency: int = 2) -> None:
        self.tokenizer = Tokenizer()
        self.min_frequency = min_frequency
        self._counter: Counter[str] = Counter()
        self._pos_map: dict[str, str] = {}
        self._category_map: dict[str, set[str]] = {}

    def _is_content_word(self, token: dict[str, str]) -> bool:
        """モジュールレベルの共通判定へ委譲する（後方互換のため残置）"""
        return is_content_word(token)

    def feed(self, text: str, category: str) -> int:
        """1つのコーパステキストを取り込む。取り込んだ語数を返す。"""
        cleaned = SPEAKER_PATTERN.sub("", text)
        cleaned = MARKDOWN_PATTERN.sub("", cleaned)
        tokens = self.tokenizer.analyze(cleaned)
        count = 0
        for token in tokens:
            if not self._is_content_word(token):
                continue
            base = token["base"]
            self._counter[base] += 1
            self._pos_map.setdefault(base, token["pos"])
            self._category_map.setdefault(base, set()).add(category)
            count += 1
        return count

    def feed_directory(self, corpus_dir: str | Path) -> int:
        """コーパスディレクトリを丸ごと取り込む"""
        corpus_path = Path(corpus_dir)
        files = sorted(p for p in corpus_path.glob("*.txt"))
        for path in files:
            category = path.stem
            text = path.read_text(encoding="utf-8")
            self.feed(text, category)
        return len(files)

    def candidates(self) -> dict[str, Candidate]:
        """頻度フィルタを通過した候補語を返す"""
        result: dict[str, Candidate] = {}
        for word, freq in self._counter.items():
            if freq < self.min_frequency:
                continue
            result[word] = Candidate(
                word=word,
                pos=self._pos_map.get(word, ""),
                frequency=freq,
                categories=sorted(self._category_map.get(word, set())),
            )
        return result

    def frequency_report(self, top_n: int = 50) -> dict:
        """頻度分布のレポート。ノイズ比率の把握に使う。"""
        total = len(self._counter)
        hapax = sum(1 for c in self._counter.values() if c == 1)
        passed = sum(1 for c in self._counter.values() if c >= self.min_frequency)
        return {
            "unique_words": total,
            "hapax_count": hapax,
            "hapax_ratio": round(hapax / total, 4) if total else 0.0,
            "passed_filter": passed,
            "min_frequency": self.min_frequency,
            "top_words": self._counter.most_common(top_n),
        }

    def save(self, path: str | Path) -> Path:
        """候補語をJSONへ保存する"""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "min_frequency": self.min_frequency,
            "report": self.frequency_report(),
            "candidates": {
                word: cand.to_dict()
                for word, cand in sorted(self.candidates().items())
            },
        }
        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return target


def load_candidates(path: str | Path) -> dict[str, Candidate]:
    """保存済み候補語ファイルを読み込む"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    result: dict[str, Candidate] = {}
    for word, payload in data.get("candidates", {}).items():
        result[word] = Candidate(
            word=word,
            pos=payload.get("pos", ""),
            frequency=int(payload.get("frequency", 0)),
            categories=list(payload.get("categories", [])),
        )
    return result