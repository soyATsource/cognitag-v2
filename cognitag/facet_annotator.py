"""
フェーズC/D: Facet 注釈と分散による検証

LLM には「語を発明する」のではなく「既に抽出された語にラベルを付ける」
仕事だけをさせる。これによりハルシネーションの入り込む余地を構造的に消す。

【離散5段階を採用する理由】
0.0〜1.0 の連続値を LLM に安定して出させるのは難しく、試行ごとの揺らぎが
そのまま分散として現れてしまう。0〜4 の整数なら JSON スキーマ強制が効き、
再現性が大きく向上する。保存時に /4 して 0.0〜1.0 に正規化する。

【分散判定が研究データそのものである点】
同一語に複数回問い、その分散を記録する。分散が閾値以下なら verified:true、
超えたら quarantine へ隔離する。この分散分布自体が
「Facet 重みは LLM によって安定的に付与できるのか」という
まだ誰も定量的に答えていない問いへの一次データになる。

【warmup trial を挟む理由（試行順序効果への対処）】
6003語の注釈で隔離された132語を分析したところ、揺れた126語の 100% で
「1回目の試行だけが外れ値、2回目と3回目は常に一致」という結果が出た。
ランダムな揺らぎなら各33%前後に散るはずであり、偶然ではありえない。
外れの方向は上下ほぼ半々（高65/低61）なので、系統的バイアスではなく
「1回目だけが不安定」という現象である。

原因は Ollama のプロンプトキャッシュと考えられる。同一プロンプトを連続で
投げると、1回目はキャッシュ無し・2回目以降はキャッシュ有りで推論され、
この状態差が出力に現れる。つまり従来の「分散」は語の多義性ではなく
Ollama のキャッシュ挙動を測っていた。05_Experiment_Design.md が警告する
疑似相関そのものである。

そこで先頭 warmup_trials 回を「捨て試行」として実行し、分散の計算からは
除外する。破棄した試行も AnnotationResult.warmup_trials に残すため、
試行順序効果そのものを後から検証できる。
warmup_trials=0 を指定すれば従来と同じ挙動になる（過去データとの比較用）。
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

from .dictionary_v2 import DictionaryV2, Entry, Provenance, now_iso
from .facets import (
    FACET_KEYS,
    LEVEL_MAX,
    annotation_notes_block,
    axis_definitions_block,
    few_shot_block,
    normalize_levels,
    valence_definition_block,
    validate_levels,
)

# 公開している dictionary_v2.json はすべてこのモデルで注釈した。
#
# より大きい gemma4:12b も試したが、VRAM 8GB の GPU には収まらず
# （8.9GB / ollama ps が 30%/70% CPU/GPU と表示する）、1回あたり
# 中央値 161 秒まで落ちた。全 6,835 語では 51 日かかる計算になる。
# 弁別力の差は残留ノイズ程度で、90 倍の時間に見合わなかった。
# 別のモデルを使う場合は --model で指定する。
DEFAULT_MODEL = "gemma3:4b"
DEFAULT_TRIALS = 3
DEFAULT_VARIANCE_THRESHOLD = 0.5  # 離散レベル(0-4)スケール上の分散
DEFAULT_WARMUP_TRIALS = 1  # 破棄する先頭試行の回数（0で従来挙動）

# Ollama の format に渡す JSON スキーマ。出力を構造的に強制する。
FACET_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {key: {"type": "integer"} for key in FACET_KEYS},
    "required": list(FACET_KEYS),
}

# 4軸と valence でスケールの意味が違うため、必ず別の節に分けて説明する。
# 同一スケールとして提示すると、valence の 0（強い不快）が
# 「無関係」と解釈され、中立語がすべて不快側に倒れる。
#
# 【語をプロンプト末尾に置く理由】
# 本プロンプトは few-shot 10例と注意書きを含み 1000 トークン近い。
# 語を先頭に置くと語ごとに接頭辞が変わり、Ollama のプロンプトキャッシュが
# 毎回破棄されて、長い静的部分を語数×試行回数だけ再計算することになる。
# VRAM に載りきらないモデルではこの再計算が支配的なコストになる。
# 静的部分を先に、可変部分（語）を最後に置けば接頭辞が共有される。
ANNOTATION_PROMPT = """語の意味を評価する作業です。

■ 第1部: 4つの側面軸
「その語がどの程度その側面を持つか」を5段階で答えます。

0 = 全く関係ない
1 = わずかに関係する
2 = ある程度関係する
3 = 強く関係する
4 = その語の本質そのもの

軸の定義:
{axes}

■ 第2部: 極性軸（スケールの意味が第1部と異なります）
「その語が快い事柄か、不快な事柄か」を5段階で答えます。
この軸では 0 は「無関係」ではありません。

0 = 強い不快（苦痛・絶望・嫌悪）
1 = やや不快
2 = 中立（快でも不快でもない。大半の語はここ）
3 = やや快
4 = 強い快（喜び・安心・感謝）

軸の定義:
{valence}

判定例:
{examples}

判定上の注意:
{notes}

出力は必ず以下のJSONのみとしてください:
{{"physical": 0-4, "psychological": 0-4, "temporal": 0-4, "logical": 0-4, "valence": 0-4}}

■ 評価する語
「{word}」
"""


@dataclass
class AnnotationResult:
    """1語分の注釈結果"""

    word: str
    trials: list[dict[str, int]] = field(default_factory=list)
    # 破棄した先頭試行の生データ。平均にも分散にも使わないが、
    # 「1回目だけが外れる」現象を後から検証するために保持する。
    warmup_trials: list[dict[str, int]] = field(default_factory=list)
    mean_levels: dict[str, float] = field(default_factory=dict)
    variances: dict[str, float] = field(default_factory=dict)
    max_variance: float = 0.0
    verified: bool = False
    error: str = ""

    @property
    def succeeded(self) -> bool:
        return bool(self.trials) and not self.error


class FacetAnnotator:
    """候補語に Facet 重みを付与する"""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        trials: int = DEFAULT_TRIALS,
        variance_threshold: float = DEFAULT_VARIANCE_THRESHOLD,
        timeout: float = 60.0,
        warmup_trials: int = DEFAULT_WARMUP_TRIALS,
    ) -> None:
        self.model = model
        self.trials = trials
        self.variance_threshold = variance_threshold
        self.timeout = timeout
        self.warmup_trials = max(0, warmup_trials)

    def _build_prompt(self, word: str) -> str:
        return ANNOTATION_PROMPT.format(
            word=word,
            axes=axis_definitions_block(),
            valence=valence_definition_block(),
            examples=few_shot_block(),
            notes=annotation_notes_block(),
        )

    def _single_trial(self, word: str) -> dict[str, int] | None:
        """1回分の注釈を取得する。失敗時は None。"""
        try:
            import ollama  # 遅延import: 解析系コマンドは ollama 無しで動かせる

            client = ollama.Client(timeout=self.timeout)
            response = client.chat(
                model=self.model,
                messages=[{"role": "user", "content": self._build_prompt(word)}],
                format=FACET_RESPONSE_SCHEMA,
                # 注釈タスクに創造性は不要。揺らぎは分散測定の邪魔にしかならない
                options={"temperature": 0.1, "top_p": 0.1},
            )
            raw = response["message"]["content"]
            parsed = json.loads(raw)
        except Exception:
            return None
        return validate_levels(parsed)

    def annotate(self, word: str) -> AnnotationResult:
        """1語を複数回注釈し、分散から検証可否を判定する。

        実行回数は warmup_trials + trials。先頭 warmup_trials 回は実行順で
        破棄し（成功・失敗にかかわらず）、残りだけで平均と分散を計算する。
        warmup は結果を使わないため、失敗しても処理は継続する。
        """
        result = AnnotationResult(word=word)

        for index in range(self.warmup_trials + self.trials):
            levels = self._single_trial(word)
            if levels is None:
                continue
            if index < self.warmup_trials:
                result.warmup_trials.append(levels)
            else:
                result.trials.append(levels)

        if not result.trials:
            result.error = "all_trials_failed"
            return result

        # 軸ごとの平均と分散を計算
        for key in FACET_KEYS:
            values = [float(t[key]) for t in result.trials]
            result.mean_levels[key] = round(statistics.fmean(values), 4)
            if len(values) >= 2:
                result.variances[key] = round(statistics.pvariance(values), 4)
            else:
                result.variances[key] = 0.0

        result.max_variance = max(result.variances.values()) if result.variances else 0.0
        # 試行が1回しか成功しなかった場合は分散を測れないので未検証扱い
        enough_trials = len(result.trials) >= 2
        result.verified = enough_trials and result.max_variance <= self.variance_threshold
        return result

    @staticmethod
    def tags_from_categories(categories: list[str]) -> list[str]:
        """出現カテゴリからタグを自動生成する。

        「素材」がキッチン家電と洗濯の両方に出現した、という情報自体が
        その語の文脈的な広がりを表す。Facet 重みが語ごとの静的な値である一方、
        タグは「どの文脈圏に属する語か」という直交した情報を保持する。
        分散が大きい語ほど複数カテゴリに跨る傾向があるかを検証する材料にもなる。
        """
        tags: list[str] = []
        for category in categories:
            tag = "#" + category.strip().replace(" ", "")
            if tag not in tags:
                tags.append(tag)
        return tags

    def to_entry(self, result: AnnotationResult, pos: str, frequency: int,
                 categories: list[str]) -> Entry:
        """注釈結果を辞書エントリへ変換する"""
        # 平均レベル(小数)を四捨五入せず、そのまま /LEVEL_MAX で正規化する
        facets = {
            key: round(result.mean_levels.get(key, 0.0) / LEVEL_MAX, 4)
            for key in FACET_KEYS
        }
        provenance = Provenance(
            source="llm_annotation",
            model=self.model,
            trials=len(result.trials),
            warmup_trials=self.warmup_trials,
            variance=result.max_variance,
            verified=result.verified,
            frequency=frequency,
            categories=categories,
            annotated_at=now_iso(),
        )
        return Entry(
            word=result.word,
            pos=pos,
            facets=facets,
            tags=self.tags_from_categories(categories),
            provenance=provenance,
        )

    def annotate_batch(
        self,
        candidates: dict,
        dictionary: DictionaryV2,
        quarantine_path: str | Path,
        interval: float = 1.0,
        save_every: int = 20,
        skip_existing: bool = True,
    ) -> dict:
        """候補語をまとめて注釈し、辞書と隔離プールへ振り分ける"""
        quarantine: dict[str, dict] = {}
        qpath = Path(quarantine_path)
        if qpath.exists():
            try:
                quarantine = json.loads(qpath.read_text(encoding="utf-8"))
            except Exception:
                quarantine = {}

        stats = {"annotated": 0, "verified": 0, "quarantined": 0, "failed": 0,
                 "skipped": 0}
        variance_log: list[float] = []

        items = sorted(candidates.items(), key=lambda kv: -kv[1].frequency)
        total = len(items)

        for index, (word, cand) in enumerate(items, 1):
            if skip_existing and dictionary.has(word):
                stats["skipped"] += 1
                continue

            print(f"[{index}/{total}] 🏷️  {word} (freq={cand.frequency})", end=" ")
            result = self.annotate(word)

            if not result.succeeded:
                stats["failed"] += 1
                print("→ ❌ 失敗")
                time.sleep(interval)
                continue

            variance_log.append(result.max_variance)
            entry = self.to_entry(result, cand.pos, cand.frequency, cand.categories)

            if result.verified:
                if dictionary.add(entry):
                    stats["verified"] += 1
                    print(f"→ ✅ var={result.max_variance}")
                else:
                    stats["failed"] += 1
                    print("→ ❌ 語として不正")
            else:
                quarantine[word] = {
                    "pos": cand.pos,
                    "frequency": cand.frequency,
                    "mean_levels": result.mean_levels,
                    "variances": result.variances,
                    "max_variance": result.max_variance,
                    "trials": result.trials,
                    # 破棄済みの先頭試行。rescue が「この語は既に warmup 済みで、
                    # trials の1件目を落とす余地はない」と判断する目印も兼ねる。
                    "warmup_trials": result.warmup_trials,
                    "warmup_trial_count": self.warmup_trials,
                }
                stats["quarantined"] += 1
                print(f"→ ⚠️ 隔離 var={result.max_variance}")

            stats["annotated"] += 1

            if stats["annotated"] % save_every == 0:
                dictionary.save()
                qpath.parent.mkdir(parents=True, exist_ok=True)
                qpath.write_text(
                    json.dumps(quarantine, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                print(f"  💾 中間保存 ({stats['annotated']}語処理済)")

            time.sleep(interval)

        dictionary.save()
        qpath.parent.mkdir(parents=True, exist_ok=True)
        qpath.write_text(
            json.dumps(quarantine, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # 分散分布のサマリ（これ自体が研究データ）
        if variance_log:
            stats["variance_summary"] = {
                "mean": round(statistics.fmean(variance_log), 4),
                "median": round(statistics.median(variance_log), 4),
                "max": round(max(variance_log), 4),
                "min": round(min(variance_log), 4),
                "threshold": self.variance_threshold,
                "n": len(variance_log),
            }
        return stats