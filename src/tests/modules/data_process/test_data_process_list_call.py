#!/usr/bin/env python3
"""DataProcessList.__call__ の動作テスト"""

from typing import Dict

from src.modules.data_process.data_process import DataProcess
from src.modules.data_process.data_process_list import DataProcessList
from src.modules.protein.protein import Protein
from src.modules.protein.protein_list import ProteinList


class _MarkerProcess(DataProcess):
    """props["marks"] にトークンを追記するだけのテスト用プロセス。"""

    def __init__(self, token: str) -> None:
        self.token = token

    def _act(self, protein: Protein) -> Protein:
        props: Dict[str, object] = protein.props
        cur = props.get("marks", "")
        props["marks"] = f"{cur}{self.token}"
        return protein


class TestDataProcessListCall:
    def test_applies_all_processes_in_order(self):
        p1 = Protein(key="p1", props={"seq": "AAAA"})
        p2 = Protein(key="p2", props={"seq": "BBBB"})
        plist = ProteinList([p1, p2])

        proc = DataProcessList([_MarkerProcess("A"), _MarkerProcess("B")])
        out = proc(plist)

        # 返り値は ProteinList（同一インスタンスが返る設計）
        assert out is plist
        assert len(out) == 2
        assert out[0].props.get("marks") == "AB"
        assert out[1].props.get("marks") == "AB"
