"""
較正テスト（Calibration Test）

facets.py に定義された各軸のアンカー語を「正解ラベル付きテストセット」として使い、
現在のプロンプト設計が軸を正しく分離できているかを定量評価する。

【なぜ必要か】
5カテゴリ・295カテゴリいずれの実行でも、軸ごとの平均重みに
  temporal 0.698 > psychological 0.641 > physical 0.536 > logical 0.326
という偏りが観測された。しかしコーパスの主題は調理器具・掃除・洗濯といった
物理的な道具と動作であり、physical が最も高くなるのが自然である。

つまりこの偏りは「コーパスの性質」ではなく「プロンプト設計」を反映している
疑いが強い。実際 few_shot 例は6例中5例で temporal が1以上であり、
LLM に「どんな語にも時間的側面が少しはある」という基準を学習させている可能性がある。

これは 05_Experiment_Design.md が警告する交絡そのものであり、
本番注釈の前に検出しておく必要がある。

【出力される指標】
  - 軸ごとの正解率: アンカー語がその軸で最高値を取った割合
  - 混同行列: どの軸がどの軸と混同されやすいか
  - 軸ごとの平均レベル: 特定の軸が一律に膨張していないか
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

from .facets import CORE_FACETS, CORE_KEYS, FACET_KEYS, LEVEL_MAX
from .facet_annotator import FacetAnnotator


class CalibrationRunner:
    """アンカー語を用いてプロンプト設計の妥当性を検証する"""

    def __init__(self, annotator: FacetAnnotator) -> None:
        self.annotator = annotator

    def run(self, interval: float = 0.0, limit_per_axis: int = 0) -> dict[str, Any]:
        results: list[dict[str, Any]] = []
        total = sum(
            len(axis.anchors[:limit_per_axis] if limit_per_axis else axis.anchors)
            for axis in CORE_FACETS
        )
        index = 0

        for axis in CORE_FACETS:
            anchors = axis.anchors[:limit_per_axis] if limit_per_axis else axis.anchors
            for word in anchors:
                index += 1
                print(f"[{index}/{total}] 🎯 {word} (期待軸: {axis.key})", end=" ")
                annotation = self.annotator.annotate(word)

                if not annotation.succeeded:
                    print("→ ❌ 失敗")
                    results.append(
                        {"word": word, "expected": axis.key, "failed": True}
                    )
                    time.sleep(interval)
                    continue

                levels = annotation.mean_levels
                # 最大軸の判定は4軸に限る。valence は「側面の強度」ではなく
                # 極性の双極スケールなので、同じ土俵で比較してはいけない。
                # 例えば「嬉しい」は psychological=4 / valence=4 になるが、
                # ここで valence を候補に入れると期待軸を外したことになってしまう。
                predicted = max(CORE_KEYS, key=lambda k: levels.get(k, 0.0))
                correct = predicted == axis.key
                mark = "✅" if correct else "❌"
                detail = " ".join(
                    f"{k[:4]}={levels.get(k, 0.0):.1f}" for k in FACET_KEYS
                )
                print(f"→ {mark} {detail}")

                results.append(
                    {
                        "word": word,
                        "expected": axis.key,
                        "predicted": predicted,
                        "correct": correct,
                        "levels": levels,
                        "variance": annotation.max_variance,
                        "failed": False,
                    }
                )
                time.sleep(interval)

        return self._summarize(results)

    def _summarize(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        valid = [r for r in results if not r.get("failed")]

        # 軸ごとの正解率
        accuracy: dict[str, dict[str, Any]] = {}
        for axis_key in CORE_KEYS:
            subset = [r for r in valid if r["expected"] == axis_key]
            hits = sum(1 for r in subset if r["correct"])
            accuracy[axis_key] = {
                "n": len(subset),
                "correct": hits,
                "rate": round(hits / len(subset), 4) if subset else 0.0,
            }

        # 混同行列: expected -> predicted の件数
        confusion: dict[str, dict[str, int]] = {
            e: {p: 0 for p in CORE_KEYS} for e in CORE_KEYS
        }
        for r in valid:
            confusion[r["expected"]][r["predicted"]] += 1

        # 軸ごとの平均レベル（一律膨張の検出）
        axis_means: dict[str, float] = {}
        for key in FACET_KEYS:
            values = [r["levels"].get(key, 0.0) for r in valid]
            axis_means[key] = (
                round(statistics.fmean(values), 4) if values else 0.0
            )

        # アンカー語が期待軸で取った平均レベル（軸の強度）
        on_axis_strength: dict[str, float] = {}
        for key in FACET_KEYS:
            subset = [r for r in valid if r["expected"] == key]
            values = [r["levels"].get(key, 0.0) for r in subset]
            on_axis_strength[key] = (
                round(statistics.fmean(values), 4) if values else 0.0
            )

        overall = (
            round(sum(1 for r in valid if r["correct"]) / len(valid), 4)
            if valid
            else 0.0
        )

        return {
            "overall_accuracy": overall,
            "n_valid": len(valid),
            "n_failed": len(results) - len(valid),
            "accuracy_by_axis": accuracy,
            "confusion_matrix": confusion,
            "axis_means_all_words": axis_means,
            "on_axis_strength": on_axis_strength,
            "details": results,
        }

    @staticmethod
    def print_report(summary: dict[str, Any]) -> None:
        print("\n" + "=" * 56)
        print(" 🎯 較正テスト結果")
        print("=" * 56)
        print(f"  全体正解率: {summary['overall_accuracy'] * 100:.1f}% "
              f"(有効{summary['n_valid']}語 / 失敗{summary['n_failed']}語)")

        print("\n--- 軸ごとの正解率 ---")
        for key, data in summary["accuracy_by_axis"].items():
            bar = "█" * int(data["rate"] * 20)
            print(f"  {key:15s} {data['correct']:2d}/{data['n']:2d} "
                  f"({data['rate'] * 100:5.1f}%) {bar}")

        print("\n--- 混同行列（行=期待, 列=予測）---")
        header = "  " + " " * 16 + "".join(f"{k[:6]:>8s}" for k in FACET_KEYS)
        print(header)
        for expected, row in summary["confusion_matrix"].items():
            cells = "".join(f"{row[p]:>8d}" for p in FACET_KEYS)
            print(f"  {expected:15s}{cells}")

        print("\n--- 全アンカー語での軸平均（一律膨張の検出）---")
        print("  ※特定軸が一律に高い場合、プロンプトのバイアスを疑う")
        for key, mean in summary["axis_means_all_words"].items():
            bar = "█" * int(mean * 8)
            print(f"  {key:15s} {mean:.2f} / {LEVEL_MAX} {bar}")

        print("\n--- 期待軸での強度（アンカー語がその軸で取った平均）---")
        print("  ※4.0に近いほど、その軸の定義が明確に伝わっている")
        for key, mean in summary["on_axis_strength"].items():
            bar = "█" * int(mean * 8)
            print(f"  {key:15s} {mean:.2f} / {LEVEL_MAX} {bar}")

    @staticmethod
    def save(summary: dict[str, Any], path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return target