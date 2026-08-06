"""
フェーズA: コーパス生成

【設計思想の転換点】
旧 expand_dict.py は LLM に「カテゴリに関連する単語を30個出せ」と指示していた。
これは LLM に語を"発明"させる行為であり、実在しない語
（例: #アトロフィックネutterstock）が混入する構造的な原因だった。

本モジュールは代わりに「そのテーマについての自然な対話文」を書かせる。
語はあくまで文脈の中に出現するものであり、後段のフェーズBで
形態素解析によって"抽出"される。LLM は語を発明できない。

副次的な利点:
  - 複数カテゴリに跨って出現する語は自然に高頻度になり、
    出現1回だけの孤立ノイズと区別可能になる（Counter が意味を持つ）
  - コーパスが残るため、Core Facets の軸定義を変更しても
    コーパスから再注釈するだけで辞書を作り直せる（感度分析の実行基盤）
"""

from __future__ import annotations

import json
import time
from pathlib import Path

DEFAULT_MODEL = "gemma3:4b"

# 生成量は「1回あたりの所要時間」に直結する。往復数を欲張ると
# タイムアウトの原因になるため、8往復程度に抑え、
# カテゴリ数を稼ぐことで総語彙量を確保する設計とする。
CORPUS_PROMPT = """「{category}」について、詳しい2人の会話を8往復書いてください。

条件:
- その分野で実際に使われる具体的な用語を会話の中で自然に使う
- 箇条書きにせず、文章として書く
- 前置きや説明は不要。対話のみ

形式:
A: （発言）
B: （発言）
"""


class CorpusBuilder:
    """カテゴリごとの対話コーパスを生成して保存する"""

    def __init__(
        self,
        out_dir: str | Path = "corpus",
        model: str = DEFAULT_MODEL,
        timeout: float = 600.0,
        temperature: float = 0.7,
        num_predict: int = 900,
        retries: int = 1,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.model = model
        self.timeout = timeout
        # 生成上限。未指定だとモデルが延々と出力を続けタイムアウトの原因になる
        self.num_predict = num_predict
        self.retries = retries
        # コーパス生成は多様な語彙を引き出したいので temperature は高めでよい
        # （注釈フェーズでは逆に 0.1 まで下げる）
        self.temperature = temperature

    def _safe_name(self, category: str) -> str:
        """カテゴリ名をファイル名に使える形へ"""
        name = category.strip().replace("/", "_").replace(" ", "_")
        name = "".join(ch for ch in name if ch not in '\\:*?"<>|')
        return name[:80]

    def path_for(self, category: str) -> Path:
        return self.out_dir / f"{self._safe_name(category)}.txt"

    def build_one(self, category: str, overwrite: bool = False) -> Path | None:
        """1カテゴリ分のコーパスを生成する。既存ならスキップ。"""
        target = self.path_for(category)
        if target.exists() and not overwrite:
            return target

        prompt = CORPUS_PROMPT.format(category=category)
        text = ""

        # タイムアウトは一時的な負荷でも起きるため、既定で1回だけ再試行する
        for attempt in range(self.retries + 1):
            started = time.time()
            try:
                import ollama  # 遅延import: extract/stats は ollama 無しで動かせる

                client = ollama.Client(timeout=self.timeout)
                response = client.chat(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    options={
                        "temperature": self.temperature,
                        "top_p": 0.9,
                        "num_predict": self.num_predict,
                        "num_ctx": 2048,
                    },
                )
                text = response["message"]["content"].strip()
                elapsed = time.time() - started
                print(f"  ⏱️ {elapsed:.1f}秒 / {len(text)}文字")
                break
            except Exception as exc:  # noqa: BLE001 - 長時間バッチなので握って続行
                elapsed = time.time() - started
                label = "再試行" if attempt < self.retries else "失敗"
                print(f"  ❌ コーパス生成{label} [{category}] "
                      f"({elapsed:.1f}秒経過): {exc}")
                text = ""

        if not text:
            return None

        if len(text) < 100:
            print(f"  ⚠️ 出力が短すぎるためスキップ [{category}] ({len(text)}文字)")
            return None

        self.out_dir.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        return target

    def warmup(self) -> bool:
        """モデルをVRAMへロードしておく。

        初回呼び出しはモデルのロード時間が加算されるため、
        バッチ開始前に一度空打ちしておくとタイムアウトを避けやすい。
        """
        print("🔥 モデルをロード中（初回は時間がかかります）...")
        started = time.time()
        try:
            import ollama

            client = ollama.Client(timeout=self.timeout)
            client.chat(
                model=self.model,
                messages=[{"role": "user", "content": "はい"}],
                options={"num_predict": 1},
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  ❌ モデルのロードに失敗しました: {exc}")
            print(f"     `ollama pull {self.model}` を実行済みか確認してください")
            return False
        print(f"  ✅ ロード完了 ({time.time() - started:.1f}秒)")
        return True

    def build_all(
        self,
        categories: list[str],
        interval: float = 5.0,
        overwrite: bool = False,
    ) -> dict[str, str]:
        """全カテゴリを順に処理する。ラズパイ想定で間隔を空ける。"""
        if not self.warmup():
            return {}

        manifest: dict[str, str] = {}
        total = len(categories)
        for index, category in enumerate(categories, 1):
            print(f"[{index}/{total}] 📝 {category}")
            path = self.build_one(category, overwrite=overwrite)
            if path is not None:
                manifest[category] = str(path)
                print(f"  ✅ {path.name}")
            time.sleep(interval)

        manifest_path = self.out_dir / "_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return manifest