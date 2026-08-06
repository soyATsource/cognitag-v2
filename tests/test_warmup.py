"""
warmup trial 方式と隔離語の救済ロジックの検証

ollama はモジュールごと差し替える。FacetAnnotator._single_trial() は
遅延 import しているため、sys.modules に偽物を入れておけばそれが使われる。
実際の LLM も Ollama サーバも不要。

実行:
  python -m pytest tests/ -v      （プロジェクト直下から）
"""

from __future__ import annotations

import json
import sys
import types

import pytest

from cognitag.facet_annotator import FacetAnnotator
from run_pipeline import recompute_without_warmup


def levels(
    physical=0, psychological=0, temporal=0, logical=0, valence=2
) -> dict[str, int]:
    """LLM の応答を模した離散レベル辞書。

    valence の既定値が 0 ではなく 2 なのは、この軸だけ双極スケールで
    2 が中立を意味するため。0 は「無関係」ではなく「強い不快」になる。
    validate_levels が全キーを要求するので、省略はできない。
    """
    return {
        "physical": physical,
        "psychological": psychological,
        "temporal": temporal,
        "logical": logical,
        "valence": valence,
    }


class FakeOllama:
    """スクリプト済みの応答を順に返す ollama モジュールの代役。

    call_count で実際の実行回数を数える。warmup 分が本当に「余分に」
    実行されているかは、これでしか確認できない。
    """

    def __init__(self, responses: list[dict[str, int] | None]) -> None:
        self.responses = responses
        self.call_count = 0
        outer = self

        class Client:
            def __init__(self, timeout: float = 60.0) -> None:
                pass

            def chat(self, **kwargs):
                index = outer.call_count
                outer.call_count += 1
                if index >= len(outer.responses):
                    raise AssertionError(
                        f"想定より多く呼ばれた: {outer.call_count}回目"
                    )
                payload = outer.responses[index]
                if payload is None:
                    raise RuntimeError("simulated trial failure")
                return {"message": {"content": json.dumps(payload)}}

        self.Client = Client


@pytest.fixture
def fake_ollama(monkeypatch):
    """responses を渡すと ollama を差し替えるファクトリを返す"""

    def install(responses):
        fake = FakeOllama(responses)
        module = types.ModuleType("ollama")
        module.Client = fake.Client
        monkeypatch.setitem(sys.modules, "ollama", module)
        return fake

    return install


# --------------------------------------------------------------------------
# warmup trial 方式
# --------------------------------------------------------------------------


def test_warmup_one_runs_four_times_and_uses_last_three(fake_ollama):
    """warmup_trials=1, trials=3 なら合計4回実行し、後半3件だけを採用する。

    1回目だけ外れ値（logical=3）にしてある。これが採用されていれば
    分散が 0 にならないので、破棄されたことが数値で確認できる。
    """
    fake = fake_ollama(
        [
            levels(physical=3, logical=3),  # 1回目: 外れ値（破棄されるはず）
            levels(physical=3, logical=1),
            levels(physical=3, logical=1),
            levels(physical=3, logical=1),
        ]
    )
    annotator = FacetAnnotator(trials=3, warmup_trials=1)

    result = annotator.annotate("機能")

    assert fake.call_count == 4
    assert len(result.trials) == 3
    assert result.trials == [levels(physical=3, logical=1)] * 3
    assert result.mean_levels["logical"] == 1.0
    assert result.max_variance == 0.0
    assert result.verified is True


def test_warmup_zero_reproduces_legacy_behaviour(fake_ollama):
    """warmup_trials=0 なら従来どおり3回実行し、1回目も採用する。

    同じ応答列で、1回目の外れ値が分散に効いてしまうことを示す。
    pvariance([3,1,1]) = 0.8889 で閾値 0.5 を超えるため隔離判定になる。
    """
    fake = fake_ollama(
        [
            levels(physical=3, logical=3),
            levels(physical=3, logical=1),
            levels(physical=3, logical=1),
        ]
    )
    annotator = FacetAnnotator(trials=3, warmup_trials=0)

    result = annotator.annotate("機能")

    assert fake.call_count == 3
    assert len(result.trials) == 3
    assert result.warmup_trials == []
    assert result.max_variance == 0.8889
    assert result.verified is False


def test_discarded_trial_is_kept_for_later_analysis(fake_ollama):
    """破棄した試行は捨てずに AnnotationResult.warmup_trials に残す"""
    fake_ollama(
        [
            levels(physical=4, logical=3),
            levels(physical=3, logical=1),
            levels(physical=3, logical=1),
            levels(physical=3, logical=1),
        ]
    )
    annotator = FacetAnnotator(trials=3, warmup_trials=1)

    result = annotator.annotate("機能")

    assert result.warmup_trials == [levels(physical=4, logical=3)]
    assert levels(physical=4, logical=3) not in result.trials


def test_failed_warmup_does_not_abort_annotation(fake_ollama):
    """warmup が失敗しても処理は継続する（結果を使わないため）"""
    fake = fake_ollama(
        [
            None,  # warmup 失敗
            levels(physical=2, temporal=2),
            levels(physical=2, temporal=2),
            levels(physical=2, temporal=2),
        ]
    )
    annotator = FacetAnnotator(trials=3, warmup_trials=1)

    result = annotator.annotate("経過")

    assert fake.call_count == 4
    assert result.succeeded is True
    assert result.warmup_trials == []  # 失敗したので生データは残らない
    assert len(result.trials) == 3
    assert result.verified is True


def test_multiple_warmup_trials_are_all_discarded(fake_ollama):
    """warmup_trials=2 なら先頭2件を破棄し、合計5回実行する"""
    fake = fake_ollama(
        [
            levels(logical=4),
            levels(logical=3),
            levels(logical=1),
            levels(logical=1),
            levels(logical=1),
        ]
    )
    annotator = FacetAnnotator(trials=3, warmup_trials=2)

    result = annotator.annotate("矛盾")

    assert fake.call_count == 5
    assert result.warmup_trials == [levels(logical=4), levels(logical=3)]
    assert result.mean_levels["logical"] == 1.0
    assert result.max_variance == 0.0


def test_provenance_records_warmup_trial_count(fake_ollama):
    """to_entry() が warmup_trials を Provenance に記録する。

    既存エントリは 0 のままなので、新旧の注釈方式を後から切り分けられる。
    """
    fake_ollama([levels(physical=4)] * 4)
    annotator = FacetAnnotator(trials=3, warmup_trials=1)

    result = annotator.annotate("包丁")
    entry = annotator.to_entry(result, pos="名詞", frequency=5, categories=["料理"])

    assert entry.provenance.warmup_trials == 1
    assert entry.provenance.trials == 3
    assert entry.to_dict()["provenance"]["warmup_trials"] == 1


# --------------------------------------------------------------------------
# rescue の再計算ロジック
# --------------------------------------------------------------------------


def test_rescue_recovers_word_whose_first_trial_was_the_outlier():
    """1回目だけが外れ、2・3回目が一致する語は分散0で救済される。

    quarantine.json の実データと同じ形（logical が 3,1,1）で検証する。
      除外前: pvariance([3,1,1]) = 0.8889 -> 閾値0.5超で隔離
      除外後: pvariance([1,1])   = 0.0    -> 救済
    """
    trials = [
        levels(physical=3, psychological=2, temporal=1, logical=3),
        levels(physical=3, psychological=2, temporal=1, logical=1),
        levels(physical=3, psychological=2, temporal=1, logical=1),
    ]

    recomputed = recompute_without_warmup(trials, warmup=1)

    assert recomputed is not None
    assert recomputed["max_variance"] == 0.0
    assert recomputed["trials"] == 2
    assert recomputed["mean_levels"]["logical"] == 1.0
    assert recomputed["mean_levels"]["physical"] == 3.0


def test_rescue_keeps_word_that_is_still_unstable():
    """2回目と3回目も食い違う語は、1回目を除いても救済されない"""
    trials = [
        levels(logical=0),
        levels(logical=4),
        levels(logical=0),
    ]

    recomputed = recompute_without_warmup(trials, warmup=1)

    assert recomputed is not None
    assert recomputed["max_variance"] == 4.0  # pvariance([4,0]) = 4.0
    assert recomputed["max_variance"] > 0.5


def test_rescue_skips_word_with_too_few_trials():
    """除外すると2件未満になる語は分散が測れないのでスキップ"""
    assert recompute_without_warmup([levels(logical=2)], warmup=1) is None
    assert (
        recompute_without_warmup([levels(logical=2), levels(logical=2)], warmup=1)
        is None
    )


def test_rescue_uses_same_statistics_as_annotator(fake_ollama):
    """再計算が FacetAnnotator.annotate() と同じ値を出す。

    計算方法がずれると、救済語と通常注釈語で分散の意味が変わってしまう。
    同じ2件の試行を、両経路に通して一致を確認する。
    """
    pair = [levels(physical=3, logical=1), levels(physical=1, logical=1)]

    fake_ollama(list(pair))
    annotator = FacetAnnotator(trials=2, warmup_trials=0)
    from_annotator = annotator.annotate("素材")

    from_rescue = recompute_without_warmup([levels(physical=0)] + pair, warmup=1)

    assert from_rescue is not None
    assert from_rescue["mean_levels"] == from_annotator.mean_levels
    assert from_rescue["variances"] == from_annotator.variances
    assert from_rescue["max_variance"] == from_annotator.max_variance
