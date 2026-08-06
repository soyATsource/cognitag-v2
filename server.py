#!/usr/bin/env python3
"""
CogniTag API サーバ（Phase 1）

辞書とスコアラを起動時に一度だけ構築し、HTTP 経由で Facet 重みを返す。
将来 Project LOGOS からこの API を叩くが、本サーバは LOGOS に依存しない。

  POST /api/facets            テキスト -> Facet 重み + coverage + token_count
  GET  /api/dictionary/stats  辞書の統計と更新日時
  GET  /api/health            死活と辞書ロード状態

起動:
  python server.py
  （uvicorn server:app --port 8010 でも可）

応答生成 /api/respond は次フェーズ。ここでは実装しない。
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from cognitag.dictionary_v2 import DictionaryV2
from cognitag.facet_scorer import FacetScorer

DICTIONARY_PATH = Path(__file__).with_name("dictionary_v2.json")
PORT = 8010


class AppState:
    """起動時に一度だけ構築し、以後リクエスト間で共有する状態。

    辞書は 4MB 超の JSON であり、リクエストごとの読み直しは論外。
    辞書ファイルが無い場合もサーバは起動する（辞書生成が未完でも
    LOGOS 側の疎通確認ができるようにするため）。
    """

    def __init__(self) -> None:
        self.dictionary = DictionaryV2(DICTIONARY_PATH)
        self.scorer: FacetScorer | None = None
        self.loaded = False
        self.load_error = ""

    def startup(self) -> None:
        if DICTIONARY_PATH.exists():
            try:
                self.dictionary.load()
                self.loaded = True
            except Exception as exc:  # 壊れた JSON でも起動は継続する
                self.load_error = f"{type(exc).__name__}: {exc}"
        else:
            self.load_error = f"dictionary not found: {DICTIONARY_PATH}"
        # 辞書が空でもスコアラは作る（全語 unknown・coverage 0.0 を返す）。
        # Tokenizer の初期化はここで一度だけ行う。
        self.scorer = FacetScorer(self.dictionary)

    @property
    def entry_count(self) -> int:
        return len(self.dictionary.entries)


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.startup()
    status = "loaded" if state.loaded else f"not loaded ({state.load_error})"
    print(f"[CogniTag] dictionary {status} / entries={state.entry_count}")
    print(f"[CogniTag] tokenizer sudachi={state.scorer.tokenizer.available}")
    yield


app = FastAPI(
    title="CogniTag API",
    version="1.0",
    description="Core Facets スコアリング API",
    lifespan=lifespan,
)

# LOGOS 側（別オリジン）から呼ぶため CORS を開ける。
#
# 【外部公開する場合は必ず絞ること】
# allow_origins=["*"] と後段の host="0.0.0.0" は、閉じた LAN 内での
# 実験を前提にした設定である。このまま インターネットへ晒すと、
# 誰でも辞書の統計とスコアリングを叩ける状態になる。
# 公開用途では allow_origins を具体的なオリジンに限定し、
# host も 127.0.0.1 にするか、前段にリバースプロキシを置くこと。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class FacetRequest(BaseModel):
    text: str = Field(..., description="スコアリング対象のテキスト")


@app.post("/api/facets")
def post_facets(req: FacetRequest) -> dict[str, Any]:
    """テキストの Facet 重みを返す。

    辞書が空でもエラーにはしない。その場合は全軸 0.0 / coverage 0.0 が返り、
    呼び出し側は coverage を見て「信頼できない」と判定できる。
    """
    assert state.scorer is not None  # lifespan で必ず構築済み
    return state.scorer.score(req.text).to_dict()


@app.get("/api/dictionary/stats")
def get_dictionary_stats() -> dict[str, Any]:
    stats = state.dictionary.stats()
    stats["updated_at"] = state.dictionary.updated_at
    stats["dictionary_loaded"] = state.loaded
    stats["path"] = str(DICTIONARY_PATH)
    return stats


@app.get("/api/health")
def get_health() -> dict[str, Any]:
    return {
        "status": "ok",
        "dictionary_loaded": state.loaded,
        "entry_count": state.entry_count,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
