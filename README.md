# CogniTag v2

日本語の単語に「物理的・心理的・時間的・論理的」の4軸で重みを持たせた辞書と、それを使って文章をスコアリングする API。

ブラックボックスではない意味解析を目指した個人プロジェクトです。文章を投げると 0.07 ミリ秒で4軸の数値が返り、どの単語がその結論に効いたかを完全に追跡できます。

## これは何か

```json
"包丁":       { "physical": 1.0, "psychological": 0.0, "temporal": 0.0, "logical": 0.0 }
"疑念":       { "physical": 0.0, "psychological": 1.0, "temporal": 0.0, "logical": 0.5 }
"したがって": { "physical": 0.0, "psychological": 0.0, "temporal": 0.0, "logical": 1.0 }
```

こうした辞書が 6,841 語ぶんあります。文章中の語をこの辞書で引き、TF-IDF で重み付けして合成します。

実際のエントリは `facets` のほかに `pos` / `tags` / `provenance`（出自と試行回数と分散）を持ちます。上は説明のために `facets` だけを抜き出したものです。

## 正直に書いておくこと

- **個人開発です。** 実務利用を想定した品質保証はしていません。
- **辞書の数値は LLM が付けたものです。** 人手で全数検証はしていません。Ollama + gemma3:4b に各語3回問い、答えが一致した語のみを採用しています。
- **コーパスも LLM が生成した対話文です。** 実在の文章ではありません。
- **検証環境が限られます。** Windows + Ollama 0.32.5 + gemma3:4b でしか動作確認していません。
- **5軸目（valence / 極性）を開発中です。** コードは対応済みですが、公開している `dictionary_v2.json` は4軸のみで極性データを持ちません。そのため API の応答に含まれる `valence` は常に `0.5`（中立）、`valence_strength` は `0.0` になります。意味のある値ではないので無視してください。
- **コーパスを拡張中です。** `cognitag/categories.py` は 966 カテゴリを定義していますが、生成済みのコーパスは 493 ファイルです。残りは `run_pipeline.py corpus` を実行すると生成されます（既存ファイルはスキップされます）。

## 品質について

正解を人手で決めたアンカー語 60 語（各軸 15 語）で精度を測っています。

```
全体の正解率 95.0%  (57/60)

physical       86.7%  (13/15)
psychological  100%   (15/15)
temporal       93.3%  (14/15)
logical        100%   (15/15)
```

詳細は `calibration_report.json` にあります。混同行列と、語ごとの判定結果が入っています。

## 必要なもの

- Python 3.11 以上（標準ライブラリの `tomllib` を使います）
- Ollama（辞書を作る場合のみ。使うだけなら不要）

### 動作確認済みのバージョン

```
Python 3.11.9
fastapi 0.139.0 / uvicorn 0.41.0 / pydantic 2.12.5
sudachipy 0.6.11 / sudachidict_core 20260723 / ollama 0.6.1

Ollama 0.32.5 / gemma3:4b / GeForce GTX 1070 Ti (VRAM 8GB)
```

## インストール

```bash
git clone https://github.com/[ユーザー名]/cognitag-v2.git
cd cognitag-v2
pip install -r requirements.txt
```

## 使い方

### API サーバを起動する

```bash
uvicorn server:app --port 8010
```

`http://127.0.0.1:8010/docs` でブラウザから試せます。

```bash
curl -X POST http://127.0.0.1:8010/api/facets \
  -H "Content-Type: application/json" \
  -d '{"text": "証拠に矛盾がある"}'
```

```json
{
  "facets": {
    "physical": 0.1033,
    "psychological": 0.6467,
    "temporal": 0.1033,
    "logical": 0.8967,
    "valence": 0.5
  },
  "coverage": 1.0,
  "matched_words": ["証拠", "矛盾"],
  "unknown_words": [],
  "unverified_words": [],
  "token_count": 2,
  "valence_strength": 0.0
}
```

`coverage` は入力語のうち辞書に存在した割合です。これが低い場合、数値の信頼度は低くなります。`matched_words` を見れば、どの語がその結論に効いたかが分かります。

`valence` と `valence_strength` は開発中の5軸目です。現在の辞書では常に固定値を返します（上の「正直に書いておくこと」を参照）。

> **注意:** `server.py` は CORS を全許可（`allow_origins=["*"]`）にし、`0.0.0.0` で待ち受けます。ローカルでの実験を想定した設定です。外部に公開する場合は必ず絞ってください。

### 辞書を自分で作り直す

```bash
python run_pipeline.py corpus     # コーパス生成（LLM使用・数時間）
python run_pipeline.py extract    # 候補語の抽出
python run_pipeline.py calibrate  # 較正テスト
python run_pipeline.py annotate   # 注釈（LLM使用・数時間）
python run_pipeline.py stats      # 状況確認
```

Windows の場合は `cognitag.bat` をダブルクリックするとメニューが出ます。日本語表示で問題が出る環境では `cognitag_ascii.bat` を使ってください。

### ある語がなぜ辞書に無いかを調べる

```bash
python check_words.py 機密 警備員 証拠
```

コーパスへの出現回数、形態素解析の結果、どのフィルタで除外されたかを表示します。

### テスト

```bash
pip install -r requirements-dev.txt
pytest
```

## 辞書の作り方

1. カテゴリごとに LLM に対話文を書かせる（語を発明させない）
2. 形態素解析して、実際に出現した語を抽出する
3. 出現2回未満の語は捨てる
4. 各語に 0〜4 の5段階で4軸の値を付けさせる。**4回聞いて1回目は捨てる**
5. 残り3回の分散が小さい語だけを採用する

4 の「1回目を捨てる」については、こちらに書きました。[記事へのリンク]

## ライセンス

- **コード**: MIT License（`LICENSE`）
- **データ**: CC BY 4.0（`LICENSE-DATA`）
  - `dictionary_v2.json` / `quarantine.json` / `calibration_report.json` / `corpus/`
