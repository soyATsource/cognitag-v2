"""
FacetScorer の検証

辞書はモックを自作する。dictionary_v2.json は生成途中で内容が変わりうるため、
実データに依存したテストは再現しない。

期待値は本文中に手計算の過程を書き、実装と同じ式を再計算しないこと。
テストが実装をなぞると、式を間違えたまま両方が一致してしまう。

実行:
  python -m pytest tests/ -v      （プロジェクト直下から）
"""

from __future__ import annotations

import math

import pytest

from cognitag.candidate_extractor import Tokenizer
from cognitag.dictionary_v2 import DictionaryV2, Entry, Provenance
from cognitag.facet_scorer import TOTAL_CATEGORIES, FacetScorer, blank_facets
from cognitag.facets import CORE_KEYS, FACET_KEYS, VALENCE_KEY, VALENCE_NEUTRAL


@pytest.fixture(scope="module")
def tokenizer() -> Tokenizer:
    """Sudachi 辞書のロードは重いので全テストで使い回す"""
    return Tokenizer()


def make_entry(
    word: str,
    facets: dict[str, float],
    category_count: int = 1,
    verified: bool = True,
) -> Entry:
    """テスト用エントリ。指定しなかった軸は既定値で埋める。

    valence だけ 0.0 ではなく中立(0.5)で埋める。双極スケールなので
    0.0 は「強い不快」を意味し、未指定の意図とずれるため。
    """
    full = {key: float(facets.get(key, 0.0)) for key in CORE_KEYS}
    full[VALENCE_KEY] = float(facets.get(VALENCE_KEY, VALENCE_NEUTRAL))
    return Entry(
        word=word,
        pos="名詞",
        facets=full,
        tags=[],
        provenance=Provenance(
            source="manual",
            trials=3,
            verified=verified,
            categories=[f"cat{i}" for i in range(category_count)],
        ),
    )


def make_scorer(tokenizer: Tokenizer, *entries: Entry) -> FacetScorer:
    dictionary = DictionaryV2("mock_dictionary.json")  # ファイルは読まない
    dictionary.entries = {entry.word: entry for entry in entries}
    return FacetScorer(dictionary, tokenizer=tokenizer)


def test_empty_dictionary_returns_zero_facets_and_zero_coverage(tokenizer):
    """辞書が空なら全軸 0.0、coverage も 0.0"""
    scorer = make_scorer(tokenizer)

    result = scorer.score("包丁で切る")

    assert result.facets == blank_facets()
    assert result.coverage == 0.0
    assert result.matched_words == []
    assert result.unknown_words == ["包丁"]
    assert result.token_count == 1


def test_weighted_average_matches_hand_calculation(tokenizer):
    """既知語のみの入力で加重平均が手計算値と一致する。

    「包丁 包丁 矛盾」
      包丁: physical=1.0, tf=2, categories=1
      矛盾: logical =1.0, tf=1, categories=1

    両語の categories 数が等しいので idf も等しく、weight の比は tf の比 2:1。
    idf を L と置くと
      physical = (2L * 1.0 + 1L * 0.0) / (2L + 1L) = 2/3 = 0.6667
      logical  = (2L * 0.0 + 1L * 1.0) / (2L + 1L) = 1/3 = 0.3333
    """
    scorer = make_scorer(
        tokenizer,
        make_entry("包丁", {"physical": 1.0}, category_count=1),
        make_entry("矛盾", {"logical": 1.0}, category_count=1),
    )

    result = scorer.score("包丁 包丁 矛盾")

    assert result.facets["physical"] == 0.6667
    assert result.facets["logical"] == 0.3333
    assert result.facets["psychological"] == 0.0
    assert result.facets["temporal"] == 0.0
    assert result.coverage == 1.0
    assert result.token_count == 3
    assert sorted(result.matched_words) == ["包丁", "矛盾"]
    assert result.unknown_words == []


def test_rare_word_outweighs_common_word_via_idf(tokenizer, monkeypatch):
    """1カテゴリにしか出ない専門語が、全カテゴリに出る汎用語より強く効く。

    「包丁と矛盾」— tf はどちらも 1 なので、差は idf のみに由来する。
      包丁: physical=1.0, categories=200 -> idf = log(295/200) ≈ 0.389
      矛盾: logical =1.0, categories=1   -> idf = log(295/1)   ≈ 5.687

    logical の取り分は 5.687/(5.687+0.389) ≈ 0.936 で physical を大きく上回る。

    文書総数を 295 に固定してから測る。ここを実際の TOTAL_CATEGORIES に
    任せると、カテゴリを1つ増やすだけで上の手計算が合わなくなり、
    実装は正しいのにテストだけが落ちる。固定値にしておけば、
    期待値の導出過程がドキュメントとして残り続ける。
    """
    monkeypatch.setattr("cognitag.facet_scorer.TOTAL_CATEGORIES", 295)

    scorer = make_scorer(
        tokenizer,
        make_entry("包丁", {"physical": 1.0}, category_count=200),
        make_entry("矛盾", {"logical": 1.0}, category_count=1),
    )

    result = scorer.score("包丁と矛盾")

    assert result.facets["logical"] > result.facets["physical"]
    assert result.facets["logical"] == pytest.approx(0.936, abs=0.005)
    assert result.facets["physical"] == pytest.approx(0.064, abs=0.005)


def test_equal_category_counts_split_evenly(tokenizer):
    """対照実験: categories 数を揃えると 50:50 になる。

    test_rare_word_outweighs_common_word_via_idf との差が
    tf ではなく idf に由来することを示す。
    """
    scorer = make_scorer(
        tokenizer,
        make_entry("包丁", {"physical": 1.0}, category_count=10),
        make_entry("矛盾", {"logical": 1.0}, category_count=10),
    )

    result = scorer.score("包丁と矛盾")

    assert result.facets["physical"] == 0.5
    assert result.facets["logical"] == 0.5


def test_word_in_every_category_has_zero_weight(tokenizer):
    """全カテゴリに出現する語は idf=0。分母が 0 になるので全軸 0.0 を返す。

    ただし辞書には在るので coverage は 1.0 のままである点に注意。
    """
    scorer = make_scorer(
        tokenizer,
        make_entry("包丁", {"physical": 1.0}, category_count=TOTAL_CATEGORIES),
    )

    result = scorer.score("包丁で切る")

    assert result.facets == blank_facets()
    assert result.coverage == 1.0
    assert result.matched_words == ["包丁"]


def test_unverified_words_are_excluded_from_calculation(tokenizer):
    """verified=False の語は Facet に一切寄与しない。

    「包丁と矛盾」で矛盾を未検証にすると、包丁のみで加重平均が決まるため
    physical=1.0 / logical=0.0 になる。矛盾は unverified_words に残る。
    """
    scorer = make_scorer(
        tokenizer,
        make_entry("包丁", {"physical": 1.0}, category_count=1),
        make_entry("矛盾", {"logical": 1.0}, category_count=1, verified=False),
    )

    result = scorer.score("包丁と矛盾")

    assert result.facets["physical"] == 1.0
    assert result.facets["logical"] == 0.0
    assert result.matched_words == ["包丁"]
    assert result.unverified_words == ["矛盾"]
    assert result.unknown_words == []
    # 辞書には存在するので coverage には数える
    assert result.coverage == 1.0
    # 除外しても内容語としては数える（交絡変数の統制用）
    assert result.token_count == 2


def test_stopwords_are_not_counted_as_tokens(tokenizer):
    """機能語・指示詞・汎用形容詞は内容語に含めない。

    「これはとてもいい包丁です」
      これ  -> 代名詞（EXCLUDED_SUBPOS）
      とても -> GENERIC_ADVERBS
      いい  -> GENERIC_ADJECTIVES かつ非自立可能
      包丁  -> 内容語
    残るのは包丁 1語のみ。
    """
    scorer = make_scorer(
        tokenizer, make_entry("包丁", {"physical": 1.0}, category_count=1)
    )

    result = scorer.score("これはとてもいい包丁です")

    assert result.token_count == 1
    assert result.matched_words == ["包丁"]
    assert result.unknown_words == []
    assert result.coverage == 1.0


def test_no_content_words_is_safe(tokenizer):
    """内容語がゼロでもゼロ除算しない"""
    scorer = make_scorer(
        tokenizer, make_entry("包丁", {"physical": 1.0}, category_count=1)
    )

    result = scorer.score("これはそうです")

    assert result.token_count == 0
    assert result.coverage == 0.0
    assert result.facets == blank_facets()


def test_facets_stay_within_unit_range(tokenizer):
    """4軸は独立。合計 1.0 への正規化は行わないが、各軸は 0.0〜1.0 に収まる。"""
    scorer = make_scorer(
        tokenizer,
        make_entry(
            "劣化",
            {"physical": 0.5, "temporal": 1.0, "logical": 0.25},
            category_count=3,
        ),
        make_entry("矛盾", {"logical": 1.0, "psychological": 0.25}, 2),
    )

    result = scorer.score("劣化と矛盾")

    assert sum(result.facets.values()) > 1.0  # 合計 1.0 に潰していない
    for key in FACET_KEYS:
        assert 0.0 <= result.facets[key] <= 1.0


def test_idf_zero_categories_does_not_divide_by_zero(tokenizer):
    """categories が空でも 1 として扱い、例外を出さない"""
    entry = make_entry("包丁", {"physical": 1.0}, category_count=1)
    entry.provenance.categories = []
    scorer = make_scorer(tokenizer, entry)

    result = scorer.score("包丁で切る")

    assert scorer.idf(entry) == pytest.approx(math.log(TOTAL_CATEGORIES))
    assert result.facets["physical"] == 1.0
