# /home/esakit25/work/tensaku/src/tensaku/al/state.py
# -*- coding: utf-8 -*-
"""
@module: tensaku.al.state
@role  : Active Learning (AL) のラベル付き / プール状態を管理するドメインモデル
@overview:
    - DatasetAdapter から得た DatasetSplit（labeled/dev/test/pool）のうち、
      特に AL で動的に変化する labeled / pool の「ID 集合」を管理する。
    - パイプライン層（pipelines.al）や sampler 層（al.sampler）からは、
      ALState を介して「どのサンプルがラベル付きか / プールか」を扱う。

@design:
    - 各サンプルには一意な ID（例: int / str）が付与されている前提。
      データアダプタ（tensaku.data.*）は、各レコードに id_key（既定 "id"）を
      含める責任を持つ。
    - ALState は「ID のリスト」を保持し、元データ本体（テキスト・スコア等）は
      DatasetSplit や DataFrame 側に保持する。
    - これにより、AL ロジックはファイルパスや DataFrame 構造に依存せず、
      純粋に ID 集合の変化だけを扱える。

@notes:
    - 現時点では単純な実装に留め、必要に応じてメタ情報（ラウンド履歴など）を
      追加していく。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Sequence, Set

from tensaku.data.base import DatasetSplit


@dataclass
class ALState:
    """
    Active Learning における「どのサンプルがどこに属しているか」を表す状態。

    Fields:
        round_index : 現在のラウンド番号（0-origin）
        labeled_ids : ラベル付きデータの ID リスト
        pool_ids    : プールデータの ID リスト
        dev_ids     : dev データの ID リスト（通常は固定）
        test_ids    : test データの ID リスト（通常は固定）

    Notes:
        - ID は int / str など hashable な型を想定する（型は Any で許容）。
        - リストの順序には意味を持たせない前提だが、必要であれば
          「先頭から優先して追加」など sampler 側の方針で解釈してよい。
    """

    round_index: int = 0
    labeled_ids: List[Any] = field(default_factory=list)
    pool_ids: List[Any] = field(default_factory=list)
    dev_ids: List[Any] = field(default_factory=list)
    test_ids: List[Any] = field(default_factory=list)

    
    

    # ----------------------------------------------------------
    # プロパティ / ヘルパ
    # ----------------------------------------------------------
    @property
    def n_labeled(self) -> int:
        return len(self.labeled_ids)

    @property
    def n_pool(self) -> int:
        return len(self.pool_ids)

    @property
    def n_dev(self) -> int:
        return len(self.dev_ids)

    @property
    def n_test(self) -> int:
        return len(self.test_ids)

    @property
    def total_n(self) -> int:
        return self.n_labeled + self.n_pool + self.n_dev + self.n_test

    @property
    def coverage(self) -> float:
        """訓練対象データに対するラベル付与割合（n_labeled / (n_labeled + n_pool)）を返す。

        NOTE:
            - dev / test は評価用のホールドアウトとして扱い、coverage の分母には含めない。
            - ラベル付きもプールも 0 のときは 0.0 を返す。
        """
        denom = self.n_labeled + self.n_pool
        if denom == 0:
            return 0.0
        return self.n_labeled / denom

    # ----------------------------------------------------------
    # 派生情報
    # ----------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """
        ログ・ダンプ用に単純な dict へ変換する。

        例:
            {
              "round_index": 3,
              "n_labeled": 120,
              "n_pool": 380,
              "n_dev": 50,
              "n_test": 100,
              "coverage": 0.18,
            }
        """
        return {
            "round_index": self.round_index,
            "n_labeled": self.n_labeled,
            "n_pool": self.n_pool,
            "n_dev": self.n_dev,
            "n_test": self.n_test,
            "coverage": self.coverage,
        }


# ======================================================================
# 初期化ユーティリティ
# ======================================================================


def _extract_ids(records: Sequence[Mapping[str, Any]], id_key: str) -> List[Any]:
    """
    レコード配列から id_key を取り出し、ID リストを返す。

    NOTE:
        - id_key が存在しないレコードはスキップする。
        - 型は問わないが hashable を推奨（ALState で集合演算に使う場合に備えて）。
    """
    ids: List[Any] = []
    for rec in records:
        if not isinstance(rec, Mapping):
            continue
        if id_key in rec:
            ids.append(rec[id_key])
    return ids


def init_state_from_split(
    split: DatasetSplit,
    id_key: str = "id",
    round_index: int = 0,
) -> ALState:
    """
    DatasetSplit から ALState を初期化する。

    Args:
        split      : DatasetAdapter から得た {labeled/dev/test/pool} 分割。
        id_key     : 各レコードの ID を表すキー名（既定 "id"）。
        round_index: 初期ラウンド番号（通常 0）。

    Returns:
        ALState: 各 split ごとの ID リストを保持する状態。
    """
    labeled_ids = _extract_ids(split.labeled, id_key=id_key)
    dev_ids = _extract_ids(split.dev, id_key=id_key)
    test_ids = _extract_ids(split.test, id_key=id_key)
    pool_ids = _extract_ids(split.pool, id_key=id_key)

    return ALState(
        round_index=round_index,
        labeled_ids=labeled_ids,
        pool_ids=pool_ids,
        dev_ids=dev_ids,
        test_ids=test_ids,
    )


# ======================================================================
# 更新ユーティリティ
# ======================================================================


def move_from_pool_to_labeled(
    state: ALState,
    selected_ids: Sequence[Any],
    as_new_round: bool = True,
) -> ALState:
    """
    pool_ids から selected_ids を取り除き、labeled_ids に追加した新しい ALState を返す。

    Args:
        state       : 現在の ALState。
        selected_ids: 新たにラベル付けしたいサンプル ID 群。
        as_new_round: True の場合、round_index を +1 した上で返す。

    Returns:
        new_state: 更新後の ALState（元の state は変更しない）。

    Notes:
        - selected_ids に pool 以外の ID が含まれていても、単に無視される
          （pool_ids に存在する ID のみ移動対象）。
        - 重複 ID は内部的に集合で処理され、labeled 側では一意な ID 集合となる。
    """
    pool_set: Set[Any] = set(state.pool_ids)
    labeled_set: Set[Any] = set(state.labeled_ids)

    moved: Set[Any] = set(selected_ids) & pool_set
    new_pool = [x for x in state.pool_ids if x not in moved]
    new_labeled = list(labeled_set | moved)

    return ALState(
        round_index=state.round_index + 1 if as_new_round else state.round_index,
        labeled_ids=new_labeled,
        pool_ids=new_pool,
        dev_ids=list(state.dev_ids),
        test_ids=list(state.test_ids),
    )


def shrink_pool(
    state: ALState,
    keep_ids: Sequence[Any],
    as_new_round: bool = False,
) -> ALState:
    """
    pool_ids を keep_ids との共通部分に絞り込んだ新しい ALState を返す。

    用途:
        - sampler 側で「候補集合」を決めたあと、それ以外の ID を pool から除外し、
          次ラウンドの母集団を制限したい場合など。

    Args:
        state    : 現在の ALState。
        keep_ids : プールとして残したい ID 群。
        as_new_round: True の場合、round_index を +1 した上で返す。

    Returns:
        new_state: 更新後の ALState。
    """
    keep_set: Set[Any] = set(keep_ids)
    new_pool = [x for x in state.pool_ids if x in keep_set]

    return ALState(
        round_index=state.round_index + 1 if as_new_round else state.round_index,
        labeled_ids=list(state.labeled_ids),
        pool_ids=new_pool,
        dev_ids=list(state.dev_ids),
        test_ids=list(state.test_ids),
    )
