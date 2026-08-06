#!/usr/bin/env python3
"""
CogniTag v2 辞書構築パイプライン

フェーズを分離して実行できる設計になっている。
ラズパイでの夜間バッチ稼働を想定し、各フェーズは中間ファイルを残すため
途中で止めても再開でき、後段だけをやり直すこともできる。

  corpus/{category}.txt   ← フェーズA（コーパス生成）
  candidates.json         ← フェーズB（頻度2以上の候補語）
  dictionary_v2.json      ← フェーズC/D（Facet付き最終辞書）
  quarantine.json         ← 分散が大きく保留された語

使い方:
  python run_pipeline.py corpus                 # フェーズA: コーパス生成
  python run_pipeline.py extract                # フェーズB: 候補語抽出
  python run_pipeline.py annotate               # フェーズC/D: Facet注釈
  python run_pipeline.py rescue                 # 隔離語の救済（再問い合わせなし）
  python run_pipeline.py stats                  # 現状のレポート表示

  python run_pipeline.py corpus --limit 5       # まず5カテゴリだけ試す
  python run_pipeline.py annotate --limit 30    # まず30語だけ注釈してみる
  python run_pipeline.py rescue --dry-run       # 救済結果を書き込まずに確認
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

from cognitag.calibration import CalibrationRunner
from cognitag.candidate_extractor import CandidateExtractor, load_candidates
from cognitag.categories import CATEGORIES
from cognitag.corpus_builder import CorpusBuilder
from cognitag.dictionary_v2 import DictionaryV2, Entry, Provenance, now_iso
from cognitag.facet_annotator import (
    DEFAULT_MODEL as DEFAULT_ANNOTATION_MODEL,
    DEFAULT_VARIANCE_THRESHOLD,
    FacetAnnotator,
)
from cognitag.facets import FACET_KEYS, LEVEL_MAX

CORPUS_DIR = Path("corpus")
CANDIDATES_PATH = Path("candidates.json")
DICTIONARY_PATH = Path("dictionary_v2.json")
QUARANTINE_PATH = Path("quarantine.json")
CALIBRATION_PATH = Path("calibration_report.json")


def cmd_corpus(args: argparse.Namespace) -> int:
    categories = CATEGORIES[: args.limit] if args.limit else CATEGORIES
    print("=" * 50)
    print(" 📝 フェーズA: コーパス生成")
    print(f" 対象カテゴリ: {len(categories)} / 全{len(CATEGORIES)}")
    print(f" モデル: {args.model}")
    print(f" タイムアウト: {args.timeout}秒 / 生成上限: {args.num_predict}トークン")
    print("=" * 50)

    builder = CorpusBuilder(
        out_dir=CORPUS_DIR,
        model=args.model,
        timeout=args.timeout,
        num_predict=args.num_predict,
        retries=args.retries,
    )
    manifest = builder.build_all(
        categories, interval=args.interval, overwrite=args.overwrite
    )
    print(f"\n✅ 完了: {len(manifest)} カテゴリ分のコーパスを生成")
    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    print("=" * 50)
    print(" 🔍 フェーズB: 候補語の抽出と頻度フィルタ")
    print("=" * 50)

    if not CORPUS_DIR.exists():
        print("❌ corpus/ がありません。先に `corpus` を実行してください。")
        return 1

    extractor = CandidateExtractor(min_frequency=args.min_frequency)
    if not extractor.tokenizer.available:
        print("⚠️ SudachiPy が見つかりません。正規表現フォールバックで動作します")
        print("   （品詞が取れないため精度が大きく落ちます。導入を強く推奨）")

    file_count = extractor.feed_directory(CORPUS_DIR)
    print(f"📂 {file_count} 件のコーパスを解析しました")

    report = extractor.frequency_report(top_n=30)
    print(f"\n--- 頻度分布 ---")
    print(f"  ユニーク語数      : {report['unique_words']}")
    print(f"  出現1回のみ(hapax): {report['hapax_count']} "
          f"({report['hapax_ratio'] * 100:.1f}%)")
    print(f"  フィルタ通過      : {report['passed_filter']} "
          f"(頻度{report['min_frequency']}以上)")
    print(f"\n--- 高頻度語 上位15 ---")
    for word, freq in report["top_words"][:15]:
        print(f"  {freq:5d}  {word}")

    path = extractor.save(CANDIDATES_PATH)
    print(f"\n✅ 候補語を保存: {path}")
    return 0


def cmd_annotate(args: argparse.Namespace) -> int:
    print("=" * 50)
    print(" 🏷️  フェーズC/D: Facet注釈と分散判定")
    print(f" モデル: {args.model} / 試行回数: {args.trials}")
    print(f" 分散閾値: {args.variance_threshold}")
    print(f" warmup試行(破棄): {args.warmup_trials}")
    print("=" * 50)

    if not CANDIDATES_PATH.exists():
        print("❌ candidates.json がありません。先に `extract` を実行してください。")
        return 1

    candidates = load_candidates(CANDIDATES_PATH)
    if args.limit:
        # 高頻度順に上位N語だけ
        items = sorted(candidates.items(), key=lambda kv: -kv[1].frequency)
        candidates = dict(items[: args.limit])
    print(f"📋 注釈対象: {len(candidates)} 語\n")

    # 出力先を差し替え可能にしてある。軸を追加した際の再注釈では、
    # 既存辞書を上書きせず新しいファイルへ書き出すこと。
    # 出力先が既にあれば skip_existing が効き、中断しても再開できる。
    dictionary_path = Path(args.dictionary)
    quarantine_path = Path(args.quarantine)
    print(f"📁 辞書  : {dictionary_path}")
    print(f"📁 隔離  : {quarantine_path}\n")
    dictionary = DictionaryV2(dictionary_path).load()
    if dictionary.entries:
        print(f"↻ 既存 {len(dictionary.entries)} 語をスキップして再開します\n")
    annotator = FacetAnnotator(
        model=args.model,
        trials=args.trials,
        variance_threshold=args.variance_threshold,
        warmup_trials=args.warmup_trials,
    )

    stats = annotator.annotate_batch(
        candidates=candidates,
        dictionary=dictionary,
        quarantine_path=quarantine_path,
        interval=args.interval,
    )

    print("\n" + "=" * 50)
    print(" 📊 注釈結果")
    print("=" * 50)
    print(f"  処理済        : {stats['annotated']}")
    print(f"  検証OK        : {stats['verified']}")
    print(f"  隔離          : {stats['quarantined']}")
    print(f"  失敗          : {stats['failed']}")
    print(f"  スキップ(既存): {stats['skipped']}")

    summary = stats.get("variance_summary")
    if summary:
        print(f"\n--- 分散分布（これ自体が研究データ）---")
        print(f"  平均   : {summary['mean']}")
        print(f"  中央値 : {summary['median']}")
        print(f"  最小   : {summary['min']}")
        print(f"  最大   : {summary['max']}")
        print(f"  閾値   : {summary['threshold']}")
        print(f"  N      : {summary['n']}")
    return 0


def recompute_without_warmup(
    trials: list[dict[str, int]], warmup: int = 1
) -> dict[str, Any] | None:
    """先頭 warmup 件を除外して平均と分散を再計算する。

    計算方法は FacetAnnotator.annotate() と同一（fmean / pvariance）でなければ
    ならない。ここがずれると、救済語と通常注釈語で分散の意味が変わってしまう。

    分散を測るには2件以上必要なので、除外後が2件未満なら None を返す。
    """
    remaining = trials[warmup:]
    if len(remaining) < 2:
        return None

    mean_levels: dict[str, float] = {}
    variances: dict[str, float] = {}
    for key in FACET_KEYS:
        values = [float(t[key]) for t in remaining]
        mean_levels[key] = round(statistics.fmean(values), 4)
        variances[key] = round(statistics.pvariance(values), 4)

    return {
        "mean_levels": mean_levels,
        "variances": variances,
        "max_variance": max(variances.values()),
        "trials": len(remaining),
    }


def cmd_rescue(args: argparse.Namespace) -> int:
    """隔離語を LLM への再問い合わせなしで救済する。

    quarantine.json には trials の生データが全て残っているため、
    1回目（キャッシュ未温存で不安定な試行）を除いて分散を再計算するだけで
    救済可否を判定できる。LLM は一切呼ばない。
    """
    print("=" * 56)
    print(" 🛟 隔離語の救済（1回目の試行を除外して再判定）")
    print(f" 分散閾値: {args.variance_threshold}")
    if args.dry_run:
        print(" ⚠️ dry-run: ファイルへの書き込みは行いません")
    print("=" * 56)

    if not QUARANTINE_PATH.exists():
        print("❌ quarantine.json がありません。")
        return 1

    quarantine: dict[str, dict] = json.loads(
        QUARANTINE_PATH.read_text(encoding="utf-8")
    )
    print(f"📋 隔離語: {len(quarantine)} 語\n")

    # categories は quarantine.json に保存されていないので候補語から引く
    candidates = (
        load_candidates(CANDIDATES_PATH) if CANDIDATES_PATH.exists() else {}
    )
    if not candidates:
        print("⚠️ candidates.json が無いため categories を補完できません。")
        print("   tags と IDF 用の categories が空のまま登録されます。")

    dictionary = DictionaryV2(DICTIONARY_PATH).load()

    rescued: dict[str, dict] = {}
    remaining_quarantine: dict[str, dict] = {}
    skipped_reasons: dict[str, str] = {}

    for word, record in quarantine.items():
        trials = record.get("trials", []) or []

        if dictionary.has(word):
            # 既存エントリは絶対に書き換えない
            remaining_quarantine[word] = record
            skipped_reasons[word] = "既に辞書にある"
            continue
        if record.get("warmup_trial_count", 0):
            # 新方式で注釈済み。先頭を落とすと正当な試行を捨てることになる
            remaining_quarantine[word] = record
            skipped_reasons[word] = "warmup適用済み"
            continue
        if len(trials) < 2:
            remaining_quarantine[word] = record
            skipped_reasons[word] = f"試行{len(trials)}件で分散を測れない"
            continue

        recomputed = recompute_without_warmup(trials, warmup=1)
        if recomputed is None:
            remaining_quarantine[word] = record
            skipped_reasons[word] = "除外後の試行が2件未満"
            continue
        if recomputed["max_variance"] > args.variance_threshold:
            remaining_quarantine[word] = record
            skipped_reasons[word] = (
                f"再計算後も分散{recomputed['max_variance']}"
            )
            continue

        categories = (
            list(candidates[word].categories) if word in candidates else []
        )
        entry = Entry(
            word=word,
            pos=record.get("pos", ""),
            facets={
                key: round(recomputed["mean_levels"][key] / LEVEL_MAX, 4)
                for key in FACET_KEYS
            },
            tags=FacetAnnotator.tags_from_categories(categories),
            provenance=Provenance(
                source="llm_annotation_rescued",
                trials=recomputed["trials"],
                warmup_trials=1,
                variance=recomputed["max_variance"],
                verified=True,
                frequency=int(record.get("frequency", 0) or 0),
                categories=categories,
                annotated_at=now_iso(),
            ),
        )
        rescued[word] = {
            "entry": entry,
            "before": record.get("max_variance", 0.0),
            "after": recomputed["max_variance"],
        }

    print(f"--- 救済対象 {len(rescued)} 語 ---")
    for word, info in sorted(rescued.items(), key=lambda kv: kv[1]["after"]):
        print(f"  ✅ {word:12s} var {info['before']} -> {info['after']}")

    print(f"\n--- 救済されなかった語 {len(remaining_quarantine)} 語 ---")
    for word in remaining_quarantine:
        print(f"  ⚠️ {word:12s} {skipped_reasons.get(word, '')}")

    if args.dry_run:
        print("\n💡 dry-run のため書き込みは行いませんでした。")
        print("   反映するには --dry-run を外して再実行してください。")
        return 0

    added = 0
    for word, info in rescued.items():
        if dictionary.add(info["entry"]):
            added += 1
        else:
            # 語として不正。隔離に戻す
            remaining_quarantine[word] = quarantine[word]
            skipped_reasons[word] = "語として不正"

    dictionary.save()
    QUARANTINE_PATH.write_text(
        json.dumps(remaining_quarantine, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 56)
    print(f" ✅ 辞書へ追加: {added} 語  (辞書計 {len(dictionary.entries)} 語)")
    print(f" ⚠️ 隔離に残留: {len(remaining_quarantine)} 語")
    print("=" * 56)
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    print("=" * 56)
    print(" 🎯 較正テスト: プロンプト設計の妥当性検証")
    print(f" モデル: {args.model} / 試行回数: {args.trials}")
    print(f" warmup試行(破棄): {args.warmup_trials}")
    print("=" * 56)
    print(" facets.py のアンカー語を正解ラベルとして、")
    print(" 各軸が正しく分離されているかを測定します。\n")

    annotator = FacetAnnotator(
        model=args.model, trials=args.trials, warmup_trials=args.warmup_trials
    )
    runner = CalibrationRunner(annotator)
    summary = runner.run(interval=args.interval, limit_per_axis=args.per_axis)
    runner.print_report(summary)

    path = runner.save(summary, CALIBRATION_PATH)
    print(f"\n💾 詳細を保存: {path}")

    acc = summary["overall_accuracy"]
    print("\n--- 判定 ---")
    if acc >= 0.75:
        print("  ✅ 良好。このまま本番注釈に進んで問題ありません。")
    elif acc >= 0.5:
        print("  ⚠️ やや不安定。混同行列で特定の軸ペアが混ざっていないか確認を。")
    else:
        print("  ❌ 分離できていません。facets.py の軸定義と few-shot 例の")
        print("     見直しを推奨します。この状態で本番を回すと")
        print("     偏った辞書が大量に生成されます。")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    print("=" * 50)
    print(" 📊 CogniTag v2 現状レポート")
    print("=" * 50)

    if CORPUS_DIR.exists():
        files = list(CORPUS_DIR.glob("*.txt"))
        total_chars = sum(len(p.read_text(encoding="utf-8")) for p in files)
        print(f"\n[コーパス] {len(files)} ファイル / 約{total_chars:,} 文字")
    else:
        print("\n[コーパス] 未生成")

    if CANDIDATES_PATH.exists():
        data = json.loads(CANDIDATES_PATH.read_text(encoding="utf-8"))
        report = data.get("report", {})
        print(f"[候補語] {len(data.get('candidates', {}))} 語 "
              f"(hapax率 {report.get('hapax_ratio', 0) * 100:.1f}%)")
    else:
        print("[候補語] 未抽出")

    if DICTIONARY_PATH.exists():
        dictionary = DictionaryV2(DICTIONARY_PATH).load()
        s = dictionary.stats()
        print(f"[辞書] 全{s['total']}語 / 検証済{s['verified']} / "
              f"未検証{s['unverified']}")
        print(f"  軸ごとの平均重み:")
        for key, mean in s["axis_means"].items():
            bar = "█" * int(mean * 30)
            print(f"    {key:15s} {mean:.3f} {bar}")
    else:
        print("[辞書] 未作成")

    if QUARANTINE_PATH.exists():
        q = json.loads(QUARANTINE_PATH.read_text(encoding="utf-8"))
        print(f"[隔離プール] {len(q)} 語")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="CogniTag v2 辞書構築パイプライン",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_corpus = sub.add_parser("corpus", help="フェーズA: コーパス生成")
    p_corpus.add_argument("--model", default="gemma3:4b")
    p_corpus.add_argument("--limit", type=int, default=0, help="先頭N件のみ処理")
    p_corpus.add_argument("--interval", type=float, default=5.0)
    p_corpus.add_argument("--overwrite", action="store_true")
    p_corpus.add_argument("--timeout", type=float, default=600.0,
                          help="1カテゴリあたりの上限秒数")
    p_corpus.add_argument("--num-predict", type=int, default=900,
                          dest="num_predict", help="生成トークン上限")
    p_corpus.add_argument("--retries", type=int, default=1,
                          help="失敗時の再試行回数")
    p_corpus.set_defaults(func=cmd_corpus)

    p_extract = sub.add_parser("extract", help="フェーズB: 候補語抽出")
    p_extract.add_argument("--min-frequency", type=int, default=2,
                           dest="min_frequency")
    p_extract.set_defaults(func=cmd_extract)

    p_annotate = sub.add_parser("annotate", help="フェーズC/D: Facet注釈")
    p_annotate.add_argument("--model", default=DEFAULT_ANNOTATION_MODEL)
    p_annotate.add_argument(
        "--dictionary", default=str(DICTIONARY_PATH),
        help="出力先の辞書。軸を追加した再注釈では新しいファイル名を指定する",
    )
    p_annotate.add_argument(
        "--quarantine", default=str(QUARANTINE_PATH),
        help="隔離プールの出力先",
    )
    p_annotate.add_argument("--trials", type=int, default=3)
    p_annotate.add_argument("--variance-threshold", type=float, default=0.5,
                            dest="variance_threshold")
    p_annotate.add_argument("--limit", type=int, default=0,
                            help="高頻度順に上位N語のみ")
    p_annotate.add_argument("--interval", type=float, default=1.0)
    p_annotate.add_argument("--warmup-trials", type=int, default=1,
                            dest="warmup_trials",
                            help="破棄する先頭試行の回数（0で無効化）")
    p_annotate.set_defaults(func=cmd_annotate)

    p_rescue = sub.add_parser(
        "rescue", help="隔離語の救済: 1回目の試行を除外して再判定（LLM再問い合わせなし）"
    )
    p_rescue.add_argument("--variance-threshold", type=float,
                          default=DEFAULT_VARIANCE_THRESHOLD,
                          dest="variance_threshold")
    p_rescue.add_argument("--dry-run", action="store_true", dest="dry_run",
                          help="書き込まず、救済される語の一覧だけ表示する")
    p_rescue.set_defaults(func=cmd_rescue)

    p_cal = sub.add_parser("calibrate", help="較正テスト: プロンプト妥当性の検証")
    p_cal.add_argument("--model", default="gemma3:4b")
    p_cal.add_argument("--trials", type=int, default=3)
    p_cal.add_argument("--interval", type=float, default=0.0)
    p_cal.add_argument("--per-axis", type=int, default=0, dest="per_axis",
                       help="各軸から先頭N語のみ使用(0=全て)")
    p_cal.add_argument("--warmup-trials", type=int, default=1,
                       dest="warmup_trials",
                       help="破棄する先頭試行の回数（0で無効化）")
    p_cal.set_defaults(func=cmd_calibrate)

    p_stats = sub.add_parser("stats", help="現状レポート")
    p_stats.set_defaults(func=cmd_stats)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())