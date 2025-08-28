#!/usr/bin/env python3
"""
SequenceContainer の機能軸テスト
"""

import pytest

from src.modules.container.sequence_container import SequenceContainer


class TestBasicOperations:
    """基本操作のテスト"""

    def test_initialization(self):
        """初期化のテスト"""
        # 空のコンテナ
        empty = SequenceContainer[int]([])
        assert len(empty) == 0
        assert empty.is_empty

        # リストから初期化
        container = SequenceContainer([1, 2, 3, 4, 5])
        assert len(container) == 5
        assert not container.is_empty

        # 他のiterableから初期化
        from_tuple = SequenceContainer((10, 20, 30))
        assert len(from_tuple) == 3
        assert list(from_tuple) == [10, 20, 30]

    def test_length_and_bool(self):
        """長さとbool変換のテスト"""
        container = SequenceContainer[str]([])
        assert len(container) == 0
        assert not bool(container)

        container.append("test")
        assert len(container) == 1
        assert bool(container)

    def test_iteration(self):
        """イテレーション機能のテスト"""
        container = SequenceContainer(["a", "b", "c"])

        # 通常のイテレーション
        result = list(container)
        assert result == ["a", "b", "c"]

        # リバースイテレーション
        reverse_result = list(reversed(container))
        assert reverse_result == ["c", "b", "a"]

    def test_membership(self):
        """メンバーシップテスト"""
        container = SequenceContainer([1, 2, 3, 4, 5])
        assert 3 in container
        assert 6 not in container
        assert "test" not in container


class TestItemAccess:
    """アイテムアクセスのテスト"""

    def test_getitem(self):
        """要素取得のテスト"""
        container = SequenceContainer(["a", "b", "c", "d", "e"])

        # インデックスアクセス
        assert container[0] == "a"
        assert container[2] == "c"
        assert container[-1] == "e"

        # スライスアクセス
        slice_result = container[1:4]
        assert isinstance(slice_result, SequenceContainer)
        assert list(slice_result) == ["b", "c", "d"]

        # ステップ付きスライス
        step_slice = container[::2]
        assert list(step_slice) == ["a", "c", "e"]

    def test_setitem(self):
        """要素設定のテスト"""
        container = SequenceContainer([1, 2, 3, 4, 5])

        # インデックス設定
        container[0] = 10
        container[2] = 30
        assert list(container) == [10, 2, 30, 4, 5]

        # スライス設定
        container[1:3] = [20, 25]
        assert list(container) == [10, 20, 25, 4, 5]

    def test_delitem(self):
        """要素削除のテスト"""
        container = SequenceContainer([1, 2, 3, 4, 5])

        # インデックス削除
        del container[2]
        assert list(container) == [1, 2, 4, 5]

        # スライス削除
        del container[1:3]
        assert list(container) == [1, 5]

    def test_index_errors(self):
        """インデックスエラーのテスト"""
        container = SequenceContainer([1, 2, 3])

        with pytest.raises(IndexError):
            _ = container[5]

        with pytest.raises(IndexError):
            container[5] = 10


class TestArithmeticOperations:
    """算術演算のテスト"""

    def test_addition(self):
        """加算演算のテスト"""
        container1 = SequenceContainer([1, 2, 3])
        container2 = SequenceContainer([4, 5, 6])

        # 通常の加算
        result = container1 + container2
        assert list(result) == [1, 2, 3, 4, 5, 6]

        # リストとの加算
        result2 = container1 + [7, 8, 9]
        assert list(result2) == [1, 2, 3, 7, 8, 9]

        # 右加算
        result3 = [0] + container1
        assert list(result3) == [0, 1, 2, 3]

    def test_inplace_addition(self):
        """インプレース加算のテスト"""
        container = SequenceContainer([1, 2, 3])
        container += [4, 5]
        assert list(container) == [1, 2, 3, 4, 5]

    def test_multiplication(self):
        """乗算演算のテスト"""
        container = SequenceContainer([1, 2])

        # 通常の乗算
        result = container * 3
        assert list(result) == [1, 2, 1, 2, 1, 2]

        # 右乗算
        result2 = 2 * container
        assert list(result2) == [1, 2, 1, 2]

    def test_inplace_multiplication(self):
        """インプレース乗算のテスト"""
        container = SequenceContainer([1, 2])
        container *= 3
        assert list(container) == [1, 2, 1, 2, 1, 2]


class TestStringRepresentation:
    """文字列表現のテスト"""

    def test_str_repr(self):
        """str/reprのテスト"""
        container = SequenceContainer([1, 2, 3])

        str_repr = str(container)
        assert "SequenceContainer" in str_repr
        assert "[1, 2, 3]" in str_repr

        repr_str = repr(container)
        assert "SequenceContainer" in repr_str
        assert "[1, 2, 3]" in repr_str


class TestModificationMethods:
    """変更メソッドのテスト"""

    def test_append(self):
        """append メソッドのテスト"""
        container = SequenceContainer[int]([])
        container.append(1)
        container.append(2)
        container.append(3)
        assert list(container) == [1, 2, 3]

    def test_insert(self):
        """insert メソッドのテスト"""
        container = SequenceContainer([1, 3, 5])
        container.insert(1, 2)
        container.insert(3, 4)
        assert list(container) == [1, 2, 3, 4, 5]

        # 範囲外のインデックス
        container.insert(10, 6)
        assert list(container) == [1, 2, 3, 4, 5, 6]

    def test_remove(self):
        """remove メソッドのテスト"""
        container = SequenceContainer([1, 2, 3, 2, 4])
        container.remove(2)  # 最初の2が削除される
        assert list(container) == [1, 3, 2, 4]

        with pytest.raises(ValueError):
            container.remove(10)  # 存在しない要素

    def test_pop(self):
        """pop メソッドのテスト"""
        container = SequenceContainer([1, 2, 3, 4, 5])

        # 最後の要素をpop
        last = container.pop()
        assert last == 5
        assert list(container) == [1, 2, 3, 4]

        # 特定のインデックスをpop
        middle = container.pop(1)
        assert middle == 2
        assert list(container) == [1, 3, 4]

        # 空のコンテナからpop
        empty = SequenceContainer[int]([])
        with pytest.raises(IndexError):
            empty.pop()

    def test_clear(self):
        """clear メソッドのテスト"""
        container = SequenceContainer([1, 2, 3, 4, 5])
        container.clear()
        assert len(container) == 0
        assert container.is_empty

    def test_extend(self):
        """extend メソッドのテスト"""
        container = SequenceContainer([1, 2])
        container.extend([3, 4, 5])
        assert list(container) == [1, 2, 3, 4, 5]

        container.extend((6, 7))
        assert list(container) == [1, 2, 3, 4, 5, 6, 7]

    def test_reverse(self):
        """reverse メソッドのテスト"""
        container = SequenceContainer([1, 2, 3, 4, 5])
        container.reverse()
        assert list(container) == [5, 4, 3, 2, 1]

    def test_sort(self):
        """sort メソッドのテスト"""
        container = SequenceContainer([3, 1, 4, 1, 5, 9, 2, 6])
        container.sort()
        assert list(container) == [1, 1, 2, 3, 4, 5, 6, 9]

        # 降順ソート
        container.sort(reverse=True)
        assert list(container) == [9, 6, 5, 4, 3, 2, 1, 1]

        # キー関数付きソート
        str_container = SequenceContainer(["apple", "pie", "cherry", "a"])
        str_container.sort(key=len)
        assert list(str_container) == ["a", "pie", "apple", "cherry"]


class TestSearchAndCount:
    """検索とカウントのテスト"""

    def test_index(self):
        """index メソッドのテスト"""
        container = SequenceContainer([1, 2, 3, 2, 4, 2])

        assert container.index(2) == 1  # 最初の出現
        assert container.index(2, 2) == 3  # start指定
        assert container.index(2, 2, 5) == 3  # start, stop指定

        with pytest.raises(ValueError):
            container.index(10)  # 存在しない要素

    def test_count(self):
        """count メソッドのテスト"""
        container = SequenceContainer([1, 2, 2, 3, 2, 4])

        assert container.count(2) == 3
        assert container.count(1) == 1
        assert container.count(10) == 0


class TestUtilityMethods:
    """ユーティリティメソッドのテスト"""

    def test_copy(self):
        """copy メソッドのテスト"""
        original = SequenceContainer([1, 2, 3])
        copied = original.copy()

        assert list(original) == list(copied)
        assert original is not copied
        assert original._data is not copied._data

        # 変更が独立していることを確認
        copied.append(4)
        assert list(original) == [1, 2, 3]
        assert list(copied) == [1, 2, 3, 4]

    def test_to_list(self):
        """to_list メソッドのテスト"""
        container = SequenceContainer([1, 2, 3])
        result = container.to_list()

        assert result == [1, 2, 3]
        assert isinstance(result, list)
        assert result is not container._data  # 別のインスタンス


class TestAdvancedOperations:
    """高度な操作のテスト"""

    def test_filter(self):
        """filter メソッドのテスト"""
        container = SequenceContainer([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        # 偶数のみフィルタ
        even = container.filter(lambda x: x % 2 == 0)
        assert list(even) == [2, 4, 6, 8, 10]

        # 5より大きいもの
        greater_than_5 = container.filter(lambda x: x > 5)
        assert list(greater_than_5) == [6, 7, 8, 9, 10]

        # 元のコンテナは変更されない
        assert list(container) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    def test_map(self):
        """map メソッドのテスト"""
        container = SequenceContainer([1, 2, 3, 4, 5])

        # 各要素を2倍
        doubled = container.map(lambda x: x * 2)
        assert list(doubled) == [2, 4, 6, 8, 10]

        # 文字列に変換
        strings = container.map(str)
        assert list(strings) == ["1", "2", "3", "4", "5"]

        # 元のコンテナは変更されない
        assert list(container) == [1, 2, 3, 4, 5]

    def test_reduce(self):
        """reduce メソッドのテスト"""
        container = SequenceContainer([1, 2, 3, 4, 5])

        # 合計
        total = container.reduce(lambda a, b: a + b)
        assert total == 15

        # 初期値付き
        total_with_initial = container.reduce(lambda a, b: a + b, 100)
        assert total_with_initial == 115

        # 最大値
        maximum = container.reduce(lambda a, b: max(a, b))
        assert maximum == 5

        # 空のコンテナでエラー
        empty = SequenceContainer[int]([])
        with pytest.raises(TypeError):
            empty.reduce(lambda a, b: a + b)


class TestSplitOperations:
    """分割操作のテスト"""

    def test_split_equal(self):
        """等分割のテスト"""
        # 基本的な等分割
        container = SequenceContainer([1, 2, 3, 4, 5, 6])
        parts = container.split_equal(3)

        assert len(parts) == 3
        assert list(parts[0]) == [1, 2]
        assert list(parts[1]) == [3, 4]
        assert list(parts[2]) == [5, 6]

        # 割り切れない場合の等分割
        container = SequenceContainer([1, 2, 3, 4, 5, 6, 7])
        parts = container.split_equal(3)

        assert len(parts) == 3
        assert list(parts[0]) == [1, 2]
        assert list(parts[1]) == [3, 4]
        assert list(parts[2]) == [5, 6, 7]  # 最後の部分に残りが全て入る

        # 単一分割
        container = SequenceContainer([1, 2, 3])
        parts = container.split_equal(1)
        assert len(parts) == 1
        assert list(parts[0]) == [1, 2, 3]

        # 空のコンテナ
        empty = SequenceContainer[int]([])
        parts = empty.split_equal(3)
        assert len(parts) == 3
        assert all(len(part) == 0 for part in parts)

        # エラーケース
        with pytest.raises(ValueError):
            container.split_equal(0)
        with pytest.raises(ValueError):
            container.split_equal(-1)

    def test_split_by_ratio(self):
        """比率分割のテスト"""
        # 基本的な比率分割
        container = SequenceContainer(list(range(1, 17)))  # [1, 2, ..., 16]
        parts = container.split_by_ratio([1, 3, 4])

        assert len(parts) == 3
        # 1:3:4の比率で16個を分割 -> 2:6:8
        assert list(parts[0]) == [1, 2]
        assert list(parts[1]) == list(range(3, 9))  # [3, 4, 5, 6, 7, 8]
        assert list(parts[2]) == list(range(9, 17))  # [9, 10, ..., 16]

        # 単一比率
        parts = container.split_by_ratio([1])
        assert len(parts) == 1
        assert list(parts[0]) == list(range(1, 17))

        # 空のコンテナ
        empty = SequenceContainer[int]([])
        parts = empty.split_by_ratio([1, 2, 3])
        assert len(parts) == 3
        assert all(len(part) == 0 for part in parts)

        # 小さなコンテナでの比率分割
        small = SequenceContainer([1, 2, 3])
        parts = small.split_by_ratio([1, 1, 1])
        assert len(parts) == 3
        assert list(parts[0]) == [1]
        assert list(parts[1]) == [2]
        assert list(parts[2]) == [3]

        # エラーケース
        with pytest.raises(ValueError):
            container.split_by_ratio([1, 0, 3])
        with pytest.raises(ValueError):
            container.split_by_ratio([1, -1, 3])


class TestEdgeCases:
    """エッジケースのテスト"""

    def test_empty_container_operations(self):
        """空のコンテナでの操作"""
        empty = SequenceContainer[int]([])

        # 各操作が適切にエラーを出すかテスト
        with pytest.raises(IndexError):
            _ = empty[0]

        with pytest.raises(IndexError):
            empty.pop()

        with pytest.raises(ValueError):
            empty.remove(1)

        assert empty.count(1) == 0
        assert empty.index.__name__ == "index"  # メソッドが存在することを確認

    def test_large_container(self):
        """大きなコンテナでの動作"""
        large_data = list(range(10000))
        container = SequenceContainer(large_data)

        assert len(container) == 10000
        assert container[9999] == 9999
        assert container.count(5000) == 1
        assert 5000 in container

    def test_type_consistency(self):
        """型の一貫性テスト"""
        int_container = SequenceContainer([1, 2, 3])
        assert isinstance(int_container[0], int)

        str_container = SequenceContainer(["a", "b", "c"])
        assert isinstance(str_container[0], str)

        # スライス結果も同じ型
        slice_result = int_container[1:3]
        assert isinstance(slice_result, SequenceContainer)


class TestJoinMethod:
    """joinメソッドのテスト"""

    def test_join_multiple_containers(self):
        """複数のコンテナを結合するテスト"""
        container1 = SequenceContainer([1, 2, 3])
        container2 = SequenceContainer([4, 5])
        container3 = SequenceContainer([6, 7, 8, 9])

        result = SequenceContainer.join([container1, container2, container3])

        assert len(result) == 9
        assert result.to_list() == [1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert isinstance(result, SequenceContainer)

    def test_join_single_container(self):
        """単一のコンテナを結合するテスト"""
        container = SequenceContainer([1, 2, 3])
        result = SequenceContainer.join([container])

        assert len(result) == 3
        assert result.to_list() == [1, 2, 3]
        assert isinstance(result, SequenceContainer)
        # 元のコンテナとは別のインスタンス
        assert result is not container

    def test_join_empty_containers(self):
        """空のコンテナを含む結合テスト"""
        container1 = SequenceContainer([1, 2])
        empty_container = SequenceContainer([])
        container2 = SequenceContainer([3, 4])

        result = SequenceContainer.join([container1, empty_container, container2])

        assert len(result) == 4
        assert result.to_list() == [1, 2, 3, 4]

    def test_join_empty_list(self):
        """空のリストを渡した場合のテスト"""
        result = SequenceContainer.join([])
        assert len(result) == 0
        assert isinstance(result, SequenceContainer)
        assert result.is_empty


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
