#!/usr/bin/env python3
"""
DataProcessList の機能軸テスト
"""

from src.modules.data_process.aggregator import Aggregator
from src.modules.data_process.data_process_list import DataProcessList


class TestBasicOperations:
    """基本操作のテスト"""

    def test_initialization_and_len(self):
        """初期化と長さの確認。"""
        plist = DataProcessList([Aggregator("mean"), Aggregator("mean")])
        assert len(plist) == 2
        assert not plist.is_empty

    def test_index_and_slice(self):
        """インデックス・スライスアクセスの確認。"""
        procs = [Aggregator("mean"), Aggregator("mean"), Aggregator("mean")]
        plist = DataProcessList(procs)

        assert isinstance(plist[0], Aggregator)
        sub = plist[1:]
        assert isinstance(sub, DataProcessList)
        assert len(sub) == 2

    def test_replace(self):
        """replace による要素入れ替え。"""
        plist = DataProcessList([Aggregator("mean")])
        out = plist.replace([Aggregator("mean"), Aggregator("mean")])

        assert out is plist
        assert len(plist) == 2
