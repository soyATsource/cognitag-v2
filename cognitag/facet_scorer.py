"""
入力テキストから Core Facets の重みを算出する。

【方式】TF-IDF 加重平均
辞書の各エントリは 4軸それぞれ 0.0〜1.0 の重みを持つ（facets.py 参照）。
テキスト全体の Facet は、出現した既知語の Facet を TF-IDF で加重平均したもの。

  weight(w) = tf(w) * idf(w)
  idf(w)    = log(TOTAL_CATEGORIES / len(provenance.categories))
  facet[ax] = SUM(weight(w) * entry.facets[ax]) / SUM(weight(w))

IDF の文書単位はコーパスのカテゴリ（295件）である。多数のカテゴリに現れる
汎用語ほど寄与が小さく、特定分野にしか現れない専門語ほど寄与が大きくなる。

【軸の正規化について】
4軸は直交していないため、合計を 1.0 に正規化してはならない（facets.py 参照）。
各 entry.facets が 0.0〜1.0 に正規化済みで、重みが非負であることから、
加重平均の結果も自動的に各軸 0.0〜1.0 に収まる。

【verified について】
provenance.verified == False の語は Facet 計算に一切含めない。未検証の語は
LLM の試行間分散が大きく、値そのものが信頼できないため。ただし
「辞書に存在はした」という事実は coverage と unverified_words に残す。

【coverage と token_count】
LOGOS 側で交絡変数を統制するための指標。
  - coverage が低い入力は、そもそも辞書が語を持っていないので Facet が不安定
  - token_count は「Facet の揺れが単に入力長と相関しているだけ」という
    対抗仮説を排除するために必要
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any

from .candidate_extractor import Tokenizer, is_content_word
from .categories import CATEGORIES
from .dictionary_v2 import DictionaryV2, Entry
from .facets import CORE_KEYS, FACET_KEYS, VALENCE_KEY, VALENCE_NEUTRAL

# IDF の文書総数。コーパスのカテゴリ数（295）と一致する。
TOTAL_CATEGORIES = len(CATEGORIES)


def blank_facets() -> dict[str, float]:
    """4軸 0.0 / valence 中立の Facet 辞書。

    valence だけ 0.0 にしてはいけない。双極スケールでは 0.0 が
    「強い不快」を意味するため、語が引けなかった入力が
    すべて不快判定になってしまう。
    """
    facets = {key: 0.0 for key in CORE_KEYS}
    facets[VALENCE_KEY] = VALENCE_NEUTRAL
    return facets


def entry_valence(entry: Entry) -> float:
    """エントリの valence を取り出す。未注釈なら中立を返す。

    valence は v3 で追加した軸であり、それ以前に注釈されたエントリは
    キーを持たない。get(..., 0.0) にすると「強い不快」になるため、
    必ずこの関数を経由すること。
    """
    return float(entry.facets.get(VALENCE_KEY, VALENCE_NEUTRAL))


@dataclass
class ScoreResult:
    """1つの入力テキストに対するスコアリング結果"""

    facets: dict[str, float]
    coverage: float
    matched_words: list[str]
    unknown_words: list[str]
    unverified_words: list[str]
    token_count: int
    # 極性の強さ 0.0〜1.0。中立からの隔たりの加重平均で、
    # 「この入力にどれだけ感情的な色が付いているか」を表す。
    # valence 自体の値は、これが小さいときは信用してはならない。
    valence_strength: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "facets": self.facets,
            "coverage": self.coverage,
            "matched_words": self.matched_words,
            "unknown_words": self.unknown_words,
            "unverified_words": self.unverified_words,
            "token_count": self.token_count,
            "valence_strength": self.valence_strength,
        }


class FacetScorer:
    """辞書を保持し、テキストを Facet 重みへ写像する"""

    def __init__(
        self, dictionary: DictionaryV2, tokenizer: Tokenizer | None = None
    ) -> None:
        self.dictionary = dictionary
        # Tokenizer の生成は Sudachi 辞書のロードを伴い重い。サーバでは
        # スコアラを使い回すこと。テスト等で共有したい場合のみ外から渡す。
        self.tokenizer = tokenizer or Tokenizer()

    def idf(self, entry: Entry) -> float:
        """カテゴリ出現数から IDF を求める。

        len(categories) が 0 の場合は 1 として扱う（ゼロ除算回避）。
        全カテゴリに出現する語の IDF は 0 となり、加重平均に寄与しない。
        破損データで categories が総数を超えた場合も IDF が負にならないよう
        上限で丸める（負の重みは加重平均を 0.0〜1.0 の外へ押し出すため）。
        """
        df = len(entry.provenance.categories) or 1
        df = min(max(df, 1), TOTAL_CATEGORIES)
        return math.log(TOTAL_CATEGORIES / df)

    def _content_words(self, text: str) -> list[str]:
        """入力テキストから内容語の見出し形を出現順に取り出す"""
        return [
            token["base"]
            for token in self.tokenizer.analyze(text)
            if is_content_word(token)
        ]

    def score(self, text: str) -> ScoreResult:
        words = self._content_words(text)
        token_count = len(words)

        # TF。異なり語の順序は初出順を保ち、応答の再現性を確保する。
        tf: Counter[str] = Counter(words)

        matched_words: list[str] = []
        unknown_words: list[str] = []
        unverified_words: list[str] = []

        totals = {key: 0.0 for key in CORE_KEYS}
        weight_sum = 0.0
        # valence は双極スケールのため4軸とは別に集計する。中立(0.5)の語を
        # 通常の加重平均に入れると、感情語が少数混じっても結果が 0.5 付近へ
        # 引き戻されて極性が消える。そこで中立からの隔たりを重みに掛け、
        # 中立語が投票に参加しないようにする。
        valence_total = 0.0
        valence_weight_sum = 0.0
        polarity_total = 0.0

        for word in tf:
            entry = self.dictionary.entries.get(word)
            if entry is None:
                unknown_words.append(word)
                continue
            if not entry.provenance.verified:
                # 辞書には在るが値が信頼できない。coverage には数えるが
                # Facet 計算からは完全に除外する。
                unverified_words.append(word)
                continue

            matched_words.append(word)
            weight = tf[word] * self.idf(entry)
            if weight <= 0.0:
                # 全カテゴリに出現する汎用語。寄与ゼロなので加算しない。
                continue
            weight_sum += weight
            for key in CORE_KEYS:
                totals[key] += weight * float(entry.facets.get(key, 0.0))

            valence = entry_valence(entry)
            polarity = abs(valence - VALENCE_NEUTRAL) * 2.0  # 0.0(中立)〜1.0(両極)
            polarity_total += weight * polarity
            valence_weight_sum += weight * polarity
            valence_total += weight * polarity * valence

        if weight_sum > 0.0:
            facets = {
                key: round(totals[key] / weight_sum, 4) for key in CORE_KEYS
            }
            if valence_weight_sum > 0.0:
                facets[VALENCE_KEY] = round(valence_total / valence_weight_sum, 4)
            else:
                # 極性を持つ語が一つもない = 感情的に中立な入力
                facets[VALENCE_KEY] = VALENCE_NEUTRAL
            valence_strength = round(polarity_total / weight_sum, 4)
        else:
            facets = blank_facets()
            valence_strength = 0.0

        distinct = len(tf)
        if distinct:
            coverage = round((distinct - len(unknown_words)) / distinct, 4)
        else:
            coverage = 0.0

        return ScoreResult(
            facets=facets,
            coverage=coverage,
            matched_words=matched_words,
            unknown_words=unknown_words,
            unverified_words=unverified_words,
            token_count=token_count,
            valence_strength=valence_strength,
        )
