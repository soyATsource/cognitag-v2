#!/usr/bin/env python3
"""
語の追跡診断ツール

「この語がなぜ辞書に入っていないのか」をパイプラインの各段階に遡って調べる。

使い方:
    python check_words.py 機密 盗む 警備員 出入り
    python check_words.py --file words.txt

各語について以下を順に確認する:
    1. 辞書 (dictionary_v2.json) に存在するか
    2. 候補語 (candidates.json) に存在するか / 頻度はいくつか
    3. コーパス (corpus/*.txt) に何回出現するか
    4. 形態素解析でどう分割されるか
    5. ストップワード・品詞フィルタに引っかかっていないか

これにより、欠落の原因が
「コーパスに出てこない」「頻度不足で切られた」「分割されている」
「フィルタで除外された」のどれかを切り分けられる。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from cognitag.candidate_extractor import (
    MARKDOWN_PATTERN,
    SPEAKER_PATTERN,
    CandidateExtractor,
    Tokenizer,
)
from cognitag.dictionary_v2 import DictionaryV2
from cognitag.stopwords import EXCLUDED_POS, EXCLUDED_SUBPOS, is_stopword

CORPUS_DIR = Path("corpus")
CANDIDATES_PATH = Path("candidates.json")
DICTIONARY_PATH = Path("dictionary_v2.json")


def load_corpus_texts() -> list[str]:
    """コーパスをファイル単位で読み込む。

    全文を結合してから解析すると Sudachi の入力上限を超えるため、
    ファイル単位で処理する（Tokenizer 側にも分割処理はあるが、
    メモリ効率の観点からもファイル単位が望ましい）。
    """
    if not CORPUS_DIR.exists():
        return []
    return [
        path.read_text(encoding="utf-8")
        for path in sorted(CORPUS_DIR.glob("*.txt"))
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="語の追跡診断")
    parser.add_argument("words", nargs="*", help="調べたい語")
    parser.add_argument("--file", help="語を1行ずつ書いたファイル")
    args = parser.parse_args()

    words = list(args.words)
    if args.file:
        words += [
            line.strip()
            for line in Path(args.file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if not words:
        parser.print_help()
        return 1

    print("=" * 60)
    print(" 🔎 語の追跡診断")
    print("=" * 60)

    # 各データソースを読み込む
    dictionary = DictionaryV2(DICTIONARY_PATH).load() if DICTIONARY_PATH.exists() else None
    print(f"[辞書]     {len(dictionary.entries) if dictionary else 0} 語")

    candidates: dict = {}
    min_freq = "?"
    if CANDIDATES_PATH.exists():
        data = json.loads(CANDIDATES_PATH.read_text(encoding="utf-8"))
        candidates = data.get("candidates", {})
        min_freq = data.get("min_frequency", "?")
    print(f"[候補語]   {len(candidates)} 語 (最低頻度 {min_freq})")

    corpus_texts = load_corpus_texts()
    corpus_text = "\n".join(corpus_texts)
    print(f"[コーパス] {len(corpus_texts)} ファイル / {len(corpus_text):,} 文字")

    tokenizer = Tokenizer()
    print(f"[解析器]   {'SudachiPy' if tokenizer.available else '正規表現フォールバック'}")

    # コーパス全体を解析して、実際の抽出結果を得る
    base_counter: Counter[str] = Counter()
    surface_counter: Counter[str] = Counter()
    pos_map: dict[str, tuple[str, str]] = {}
    print("  コーパスを解析中...", end="", flush=True)
    for text in corpus_texts:
        cleaned = MARKDOWN_PATTERN.sub("", SPEAKER_PATTERN.sub("", text))
        if not cleaned:
            continue
        for t in tokenizer.analyze(cleaned):
            base_counter[t["base"]] += 1
            surface_counter[t["surface"]] += 1
            pos_map.setdefault(t["base"], (t["pos"], t.get("subpos", "*")))
    print(" 完了")

    for word in words:
        print("\n" + "-" * 60)
        print(f" 📌 {word}")
        print("-" * 60)

        # 1. 辞書
        if dictionary and dictionary.has(word):
            entry = dictionary.entries[word]
            facets = " ".join(f"{k[:4]}={v:.2f}" for k, v in entry.facets.items())
            print(f"  ✅ 辞書に存在  {facets}")
            print(f"     頻度={entry.provenance.frequency} "
                  f"分散={entry.provenance.variance} "
                  f"カテゴリ数={len(entry.provenance.categories)}")
            continue
        print("  ❌ 辞書に無し")

        # 2. 候補語
        if word in candidates:
            c = candidates[word]
            print(f"  ⚠️ 候補語には存在（頻度{c.get('frequency')}）"
                  f" → 注釈が未実行か失敗している")
            continue
        print("  ❌ 候補語にも無し")

        # 3. コーパス中の出現
        raw_hits = corpus_text.count(word)
        base_hits = base_counter.get(word, 0)
        surf_hits = surface_counter.get(word, 0)
        print(f"  コーパス内の文字列出現: {raw_hits} 回")
        print(f"  形態素の原形として    : {base_hits} 回")
        print(f"  形態素の表層形として  : {surf_hits} 回")

        if raw_hits == 0:
            print("  → 原因: コーパスにそもそも出現していない")
            print("     この語を含むカテゴリを categories.py に追加する必要がある")
            continue

        if base_hits == 0:
            # 分割されている可能性を調べる
            sub = tokenizer.analyze(word)
            pieces = " + ".join(t["base"] for t in sub)
            print(f"  → 原因: 形態素解析で分割されている（{word} → {pieces}）")
            for t in sub:
                mark = "除外" if (
                    t["pos"] in EXCLUDED_POS
                    or t.get("subpos") in EXCLUDED_SUBPOS
                    or is_stopword(t["base"])
                ) else "採用"
                print(f"       {t['base']:10s} {t['pos']}/{t.get('subpos')} → {mark}")
            continue

        # 4. フィルタ判定
        pos, subpos = pos_map.get(word, ("?", "?"))
        print(f"  品詞: {pos} / {subpos}")
        reasons = []
        if pos in EXCLUDED_POS:
            reasons.append(f"品詞 {pos} が除外対象")
        if subpos in EXCLUDED_SUBPOS:
            reasons.append(f"細分類 {subpos} が除外対象")
        if is_stopword(word):
            reasons.append("ストップワードに登録されている")
        if len(word) < 2:
            reasons.append("1文字のため除外")
        if len(word) == 2 and all("ぁ" <= ch <= "ん" for ch in word):
            reasons.append("ひらがな2文字のため除外")
        if isinstance(min_freq, int) and base_hits < min_freq:
            reasons.append(f"頻度 {base_hits} が最低頻度 {min_freq} 未満")

        if reasons:
            print("  → 原因:")
            for r in reasons:
                print(f"       - {r}")
        else:
            print("  → 原因不明。extract を再実行していない可能性がある")

    print("\n" + "=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())