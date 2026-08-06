"""
Core Facets の定義（DVF-Arch の基底軸）

このモジュールが Core Facets の「操作的定義」そのものである。
軸の定義文と few-shot アンカー例が、実質的に注釈の基準を決定するため、
ここを変更した場合は必ず辞書を再注釈すること（corpus からやり直す必要はない）。

設計上の注意:
- 4軸は直交していない。「殴る」は physical と psychological の両方に重みを持つ。
- したがって合計を 1.0 に正規化してはならない。各軸独立の 0.0〜1.0 とする。
- LLM への問いかけは 0〜4 の離散5段階。保存時に /4 して正規化する。
"""

from __future__ import annotations

from dataclasses import dataclass

# 離散化の段階数（0..LEVEL_MAX）
LEVEL_MAX = 4


@dataclass(frozen=True)
class FacetAxis:
    """1つの Core Facet 軸の定義"""

    key: str
    label_ja: str
    definition: str  # LLM に提示する軸の定義文
    anchors: tuple[str, ...]  # 人手選定したアンカー語（この軸が強い代表例）


CORE_FACETS: tuple[FacetAxis, ...] = (
    FacetAxis(
        key="physical",
        label_ja="物理",
        definition="物質・空間・身体・目に見える対象や、それらへの作用に関わるか",
        anchors=(
            "包丁", "石", "壁", "殴る", "重い", "液体",
            "温度", "移動する", "場所", "破壊する",
            "ざらざら", "身震い", "骨", "エンジン", "標識",
        ),
    ),
    FacetAxis(
        key="psychological",
        label_ja="心理",
        definition="感情・動機・信念・主観的な状態や、その変化に関わるか",
        anchors=(
            "嬉しい", "疑念", "不安", "嫉妬", "期待", "諦め",
            "恥ずかしい", "共感", "信じる", "ためらう",
            "動機", "気質", "好意", "嫌悪", "安心",
        ),
    ),
    FacetAxis(
        key="temporal",
        label_ja="時間",
        definition="時間の流れ・順序・変化・持続・頻度に関わるか",
        anchors=(
            "徐々に", "突然", "しばしば", "最初", "最後", "途中",
            "経過", "持続する", "一瞬", "周期", "劣化", "成長",
            "以前", "その後", "同時に",
        ),
    ),
    FacetAxis(
        key="logical",
        label_ja="論理",
        definition="因果・推論・条件・矛盾・真偽など、論理構造に関わるか",
        anchors=(
            "したがって", "ゆえに", "にもかかわらず", "すなわち", "ただし",
            "もし", "矛盾", "根拠", "仮定", "帰結",
            "証明する", "否定する", "条件", "必然", "反証",
        ),
    ),
)

# --- valence（快・不快） -------------------------------------------------
#
# 【なぜ4軸と分けて定義するか】
# 4軸は「その側面をどの程度持つか」を測る強度スケールで、0 は「無関係」を意味する。
# valence だけはスケールの意味が違い、0 が「強い不快」、2 が「中立」、4 が「強い快」
# という双極スケールである。0 が「無関係」ではない点が決定的に異なるため、
# CORE_FACETS には入れず、プロンプト上も別建てで説明する。
#
# 【必要になった理由】
# 実対話ログの較正で、「嬉しい」と「不安」が4軸上では区別できないことが判明した。
# どちらも psychological が高いだけで、共感すべきか祝福すべきかが決まらない。
# 応答方針を規則で決めるには極性が不可欠である。
VALENCE_KEY = "valence"

VALENCE_AXIS = FacetAxis(
    key=VALENCE_KEY,
    label_ja="極性",
    definition=(
        "その語が快い事柄か不快な事柄か（0=強い不快, 2=中立, 4=強い快）。"
        "他の4軸と違い、0 は「無関係」ではなく「強い不快」を意味する"
    ),
    anchors=(
        "嬉しい", "感謝", "安心", "祝う", "成功",
        "不安", "嫌悪", "失敗", "苦痛", "絶望",
    ),
)

# 正規化後の中立値。辞書に valence を持たない旧エントリは、
# 0.0（=強い不快）ではなくこの値として扱わなければならない。
VALENCE_NEUTRAL = 0.5

# 4軸のみのキー。最大軸の判定やコントラスト制約は、意味の異なる valence を
# 混ぜてはいけないので、必ずこちらを使うこと。
CORE_KEYS: tuple[str, ...] = tuple(axis.key for axis in CORE_FACETS)

# 保存・注釈・検証で扱う全キー（4軸 + valence）
FACET_KEYS: tuple[str, ...] = CORE_KEYS + (VALENCE_KEY,)

# few-shot 例。ここが注釈の基準を最も強く規定する。
#
# 【v2 での修正理由】
# 較正テスト(60語)により、以下の系統的バイアスが検出された:
#   全アンカー語での軸平均  temporal 2.80 > psychological 2.54
#                          > logical 1.95 > physical 1.22
#   logical のアンカー語のうち3語が temporal と誤判定された
#
# 原因は旧 few-shot が6例中5例で temporal >= 1 だったこと。
# 「どんな語にも時間的側面が少しはある」という基準を LLM に学習させ、
# 結果として temporal が膨張し、logical を吸収していた。
# 逆に physical は「殴る」等の動作例に偏り、物体・物質への適用が抑制された。
#
# 【修正方針】
#   1. temporal = 0 の例を多数派にする（時間性は明示的な語のみ）
#   2. 物体・物質で physical が高い例を増やす（動作以外にも物理性を認めさせる）
#   3. 順接の接続詞を logical=4 / temporal=0 と明示し、両軸を分離する
#   4. 多軸に跨る例を1つ残し、軸が独立であることを示す
#
# 【v3 での修正理由】
# gemma3:4b による 6,841 語の注釈を実対話 151 発話で較正したところ、
# 全4軸が 0.2〜0.8 の中庸に収まる語が 41% を占め、軸がほとんど分離しなかった。
# 最大軸と2位の差の中央値は 0.061 しかなく、応答方針の決定に使えない。
# 試行間分散は平均 0.013（p90 は 0.0）なので、モデルは「一貫して中庸な値」を
# 出していた。つまり再現性は高いが弁別力がない。
#
# 対策は2つ。
#   1. 全 few-shot 例で「0 を付ける軸」と「3以上を付ける軸」を必ず共存させ、
#      中庸な回答が例に一つも存在しない状態にする（下記は全例がこれを満たす）
#   2. ANNOTATION_NOTES にコントラスト制約を明文化する
FEW_SHOT_EXAMPLES: tuple[tuple[str, dict[str, int]], ...] = (
    # 物体そのもの: 動作でなくても physical は高い。道具なので valence は中立
    ("包丁", {"physical": 4, "psychological": 0, "temporal": 0, "logical": 0,
              "valence": 2}),
    # 物質: 物体と同様に physical。temporal は 0
    ("洗剤", {"physical": 3, "psychological": 0, "temporal": 0, "logical": 0,
              "valence": 2}),
    # 物理的動作: 行為に時間がかかることを理由に temporal を上げない
    ("殴る", {"physical": 4, "psychological": 2, "temporal": 0, "logical": 0,
              "valence": 0}),
    # 論理接続: 順序ではなく論理。temporal を 0 にして両軸を分離する
    ("したがって", {"physical": 0, "psychological": 0, "temporal": 0, "logical": 4,
                    "valence": 2}),
    # 論理概念: 時間と無関係
    ("矛盾", {"physical": 0, "psychological": 1, "temporal": 0, "logical": 4,
              "valence": 1}),
    # 心理状態: 感情に持続時間があることを理由に temporal を上げない
    ("疑念", {"physical": 0, "psychological": 4, "temporal": 0, "logical": 2,
              "valence": 1}),
    # 真に時間的な語: これが temporal=4 の基準
    ("徐々に", {"physical": 0, "psychological": 0, "temporal": 4, "logical": 0,
                "valence": 2}),
    # 多軸に跨る例: 軸が独立であることを示す
    ("劣化", {"physical": 2, "psychological": 0, "temporal": 4, "logical": 1,
              "valence": 1}),
    # --- valence を追加した理由そのものを示す対の例 ---
    # 「嬉しい」と「不安」は4軸上ではほぼ同一（psychological が高いだけ）。
    # valence だけが両者を分ける。この対を必ず両方 few-shot に置くこと。
    ("嬉しい", {"physical": 0, "psychological": 4, "temporal": 0, "logical": 0,
                "valence": 4}),
    ("不安", {"physical": 0, "psychological": 4, "temporal": 0, "logical": 1,
              "valence": 0}),
)

# 軸ごとの判定上の注意。較正テストで検出された誤りパターンを直接是正する。
ANNOTATION_NOTES: tuple[str, ...] = (
    "temporal は「時間の経過・順序・変化」を語そのものが明示的に含む場合のみ値を付けます。"
    "「行為には時間がかかる」「状態には持続がある」という理由で値を付けてはいけません。",
    "physical は動作だけでなく、物体・物質・道具・身体部位・材質にも高い値を付けます。",
    "順接の接続詞（したがって、ゆえに、そこで）は logical であり temporal ではありません。"
    "時間的順序ではなく論理的帰結を表すためです。",
    "4つの軸は独立です。どの軸も低い語があって構いません。無理に値を分散させないでください。",
    # v3 追加。gemma3:4b は全軸に 1〜2 を付ける傾向が強く、辞書の 41% が
    # 弁別力を失っていた。これを構造的に禁止する。
    "physical / psychological / temporal / logical の4軸については、"
    "必ず1つ以上の軸に 0 を、1つ以上の軸に 3 以上を付けてください。"
    "4軸すべてを 1〜2 にする回答は禁止です。判断に迷う場合も、"
    "最も関係の深い軸と最も関係の薄い軸を選び、はっきり差をつけてください。",
    "valence はこのコントラスト規則の対象外です。"
    "快でも不快でもない語には迷わず 2（中立）を付けてください。",
)


def blank_levels() -> dict[str, int]:
    """全軸 0 の離散レベル辞書を返す"""
    return {key: 0 for key in FACET_KEYS}


def normalize_levels(levels: dict[str, int]) -> dict[str, float]:
    """離散レベル(0..4)を 0.0〜1.0 の Facet 重みへ変換する。

    合計を1.0にする正規化は行わない（軸は直交していないため）。
    """
    weights: dict[str, float] = {}
    for key in FACET_KEYS:
        raw = levels.get(key, 0)
        try:
            value = int(raw)
        except (TypeError, ValueError):
            value = 0
        value = max(0, min(LEVEL_MAX, value))
        weights[key] = round(value / LEVEL_MAX, 4)
    return weights


def validate_levels(levels: object) -> dict[str, int] | None:
    """LLM 応答が正しい離散レベル辞書か検証する。不正なら None。"""
    if not isinstance(levels, dict):
        return None
    cleaned: dict[str, int] = {}
    for key in FACET_KEYS:
        if key not in levels:
            return None
        try:
            value = int(levels[key])
        except (TypeError, ValueError):
            return None
        if not (0 <= value <= LEVEL_MAX):
            return None
        cleaned[key] = value
    return cleaned


def axis_definitions_block() -> str:
    """プロンプトに埋め込む4軸の定義ブロックを生成する。

    valence はスケールの意味が異なるため、ここには含めない
    （valence_definition_block を別に埋め込むこと）。
    """
    lines = []
    for axis in CORE_FACETS:
        lines.append(f"- {axis.key}: {axis.definition}")
    return "\n".join(lines)


def valence_definition_block() -> str:
    """プロンプトに埋め込む valence の定義ブロックを生成する"""
    return f"- {VALENCE_AXIS.key}: {VALENCE_AXIS.definition}"


def annotation_notes_block() -> str:
    """プロンプトに埋め込む判定上の注意ブロックを生成する"""
    return "\n".join(f"- {note}" for note in ANNOTATION_NOTES)


def few_shot_block() -> str:
    """プロンプトに埋め込む few-shot 例ブロックを生成する"""
    import json

    lines = []
    for word, levels in FEW_SHOT_EXAMPLES:
        lines.append(f'「{word}」 -> {json.dumps(levels, ensure_ascii=False)}')
    return "\n".join(lines)