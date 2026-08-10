"""
較正レポートの整形のテスト。

LLM は呼ばない。_summarize が作った集計をそのまま print_report に
通せることだけを見る。

このテストがある理由:
混同行列を4軸で作るよう変えたのに、表示側が5軸を参照したままだった。
集計は正しく終わっていたのに、表示の直前で KeyError になり、
その後の save まで到達せず、実行結果がまるごと失われた。
7分かけた測定が表示のバグで消えるのは割に合わない。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cognitag.calibration import CalibrationRunner  # noqa: E402
from cognitag.facets import CORE_KEYS, FACET_KEYS  # noqa: E402


def make_results() -> list[dict]:
    """4軸それぞれ1語ずつ、正解と不正解を1件ずつ含む集計対象。"""
    results = []
    for axis in CORE_KEYS:
        levels = {key: 0.0 for key in FACET_KEYS}
        levels[axis] = 4.0
        levels["valence"] = 2.0
        results.append(
            {
                "word": f"{axis}語",
                "expected": axis,
                "predicted": axis,
                "correct": True,
                "levels": levels,
                "variance": 0.0,
                "failed": False,
            }
        )
    # 1件だけ外す（physical を期待したが temporal になった）
    wrong = dict(results[0])
    wrong["word"] = "移動する"
    wrong["predicted"] = "temporal"
    wrong["correct"] = False
    results.append(wrong)
    return results


@pytest.fixture
def summary() -> dict:
    runner = CalibrationRunner(annotator=None)
    return runner._summarize(make_results())


class Test集計:
    def test_混同行列は4軸のみ(self, summary: dict):
        # valence は極性の双極スケールで、最大軸の予測先にはなりえない
        assert set(summary["confusion_matrix"]) == set(CORE_KEYS)
        for row in summary["confusion_matrix"].values():
            assert set(row) == set(CORE_KEYS)

    def test_軸ごとの正解率も4軸のみ(self, summary: dict):
        assert set(summary["accuracy_by_axis"]) == set(CORE_KEYS)

    def test_軸平均にはvalenceも含む(self, summary: dict):
        # 一律膨張の検出には valence の平均も見たい
        assert set(summary["axis_means_all_words"]) == set(FACET_KEYS)


class Test表示:
    def test_print_reportが例外を出さない(self, summary: dict, capsys):
        """ここが落ちると、集計が終わっていても save まで到達しない。"""
        runner = CalibrationRunner(annotator=None)
        runner.print_report(summary)

        out = capsys.readouterr().out
        assert "混同行列" in out
        for axis in CORE_KEYS:
            assert axis[:6] in out

    def test_保存できる(self, summary: dict, tmp_path: Path):
        runner = CalibrationRunner(annotator=None)
        path = runner.save(summary, tmp_path / "report.json")
        assert Path(path).exists()
