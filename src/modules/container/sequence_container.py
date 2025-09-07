from __future__ import annotations

"""
シーケンス要素を安全に扱うための汎用コンテナ基底クラス。

- list 互換の操作（追加・削除・結合・スライス等）を提供します。
- スライスや結合でも同型（サブクラス）を保つように設計されています。
"""

from typing import Any, Callable, Iterable, Iterator, Optional, Self, overload


class SequenceContainer[T]:
    """list 互換の操作を型安全に提供する汎用シーケンスコンテナ。"""

    def __init__(self, iterable: Iterable[T]) -> None:
        self._data = list(iterable)

    # === Basic Properties ===
    def __len__(self) -> int:
        """要素数を返す。"""
        return len(self._data)

    def __bool__(self) -> bool:
        """空でなければ True。"""
        return bool(self._data)

    @property
    def is_empty(self) -> bool:
        """空なら True。"""
        return len(self._data) == 0

    # === Iteration ===
    def __iter__(self) -> Iterator[T]:
        """要素のイテレータを返す。"""
        return iter(self._data)

    def __reversed__(self) -> Iterator[T]:
        """逆順イテレータを返す。"""
        return reversed(self._data)

    # === Membership ===
    def __contains__(self, item: T) -> bool:
        """要素が含まれていれば True。"""
        return item in self._data

    # === Item Access ===
    @overload
    def __getitem__(self, i: int) -> T:
        pass

    @overload
    def __getitem__(self, s: slice) -> Self:
        pass

    def __getitem__(self, key: int | slice) -> T | Self:
        """指定位置もしくはスライスを取得する。

        スライス時はサブクラスの ``__init__`` を呼ばずに、
        同型インスタンスをシャローコピーして ``_data`` だけ差し替える。
        これにより、``Dataloader`` のように特殊な ``__init__``
        （例: 設定オブジェクト必須）を持つサブクラスでも安全にスライス可能。
        """
        if isinstance(key, slice):
            new = object.__new__(type(self))
            # __dict__ を持つ場合はコピーしてメタ情報を引き継ぐ
            if hasattr(self, "__dict__"):
                new.__dict__ = self.__dict__.copy()  # type: ignore[attr-defined]
            # _data のみスライス結果に置換
            new._data = self._data[key]  # type: ignore[attr-defined]
            return new
        return self._data[key]

    def __setitem__(self, key: int | slice, value: T | Iterable[T]) -> None:
        """指定位置またはスライスに値を設定する。"""
        if isinstance(key, slice):
            if not isinstance(value, Iterable):
                raise TypeError("can only assign an iterable to a slice")
            self._data[key] = list(value)
        else:
            # For single index assignment, value should be T, not Iterable[T]
            # We'll cast to T to satisfy type checker
            self._data[key] = value  # type: ignore[assignment]

    def __delitem__(self, key: int | slice) -> None:
        """指定位置またはスライスを削除する。"""
        del self._data[key]

    # === Arithmetic Operations ===
    def __add__(self, other: Iterable[T]) -> Self:
        """自身と引数を連結した新しい同型コンテナを返す。"""
        return type(self)(self._data + list(other))

    def __radd__(self, other: Iterable[T]) -> Self:
        """引数を前、自身を後に連結した新しい同型コンテナを返す。"""
        return type(self)(list(other) + self._data)

    def __iadd__(self, other: Iterable[T]) -> Self:
        """自身を引数要素で拡張し、自己を返す（破壊的）。"""
        self.extend(other)
        return self

    def __mul__(self, n: int) -> Self:
        """要素を n 回繰り返した新しい同型コンテナを返す。"""
        return type(self)(self._data * n)

    def __rmul__(self, n: int) -> Self:
        """要素を n 回繰り返した新しい同型コンテナを返す（右側演算）。"""
        return type(self)(n * self._data)

    def __imul__(self, n: int) -> Self:
        """要素を n 回繰り返し、自己を返す（破壊的）。"""
        self._data *= n
        return self

    # === String Representation ===
    def __str__(self) -> str:
        """簡易な文字列表現を返す。"""
        return f"SequenceContainer({self._data})"

    def __repr__(self) -> str:
        """詳細な文字列表現を返す。"""
        return f"SequenceContainer({self._data!r})"

    # === Modification Operations ===
    def append(self, item: T) -> None:
        """末尾に要素を追加する。"""
        self._data.append(item)

    def insert(self, index: int, item: T) -> None:
        """指定位置に要素を挿入する。"""
        self._data.insert(index, item)

    def remove(self, item: T) -> None:
        """最初の一致要素を削除する。"""
        self._data.remove(item)

    def pop(self, index: int = -1) -> T:
        """指定位置の要素を取り除き返す（既定は末尾）。"""
        return self._data.pop(index)

    def clear(self) -> None:
        """全要素を削除する。"""
        self._data.clear()

    def extend(self, iterable: Iterable[T]) -> None:
        """イテラブルの要素を末尾に追加する。"""
        self._data.extend(iterable)

    def reverse(self) -> None:
        """要素順を反転する（破壊的）。"""
        self._data.reverse()

    def sort(self, *, key: Optional[Callable[[T], Any]] = None, reverse: bool = False) -> None:
        """要素を並べ替える（破壊的）。"""
        self._data.sort(key=key, reverse=reverse)

    # === Search and Count Operations ===
    def index(self, item: T, start: int = 0, stop: Optional[int] = None) -> int:
        """最初に出現する位置を返す。範囲を指定可能。"""
        if stop is None:
            return self._data.index(item, start)
        return self._data.index(item, start, stop)

    def count(self, item: T) -> int:
        """要素の出現回数を返す。"""
        return self._data.count(item)

    # === Utility Methods ===
    def copy(self) -> Self:
        """浅いコピーを返す。"""
        return type(self)(self._data)

    def to_list(self) -> list[T]:
        """内部リストのコピーを返す。"""
        return self._data.copy()

    def replace(self, iterable: Iterable[T]) -> Self:
        """内部データ ``_data`` を与えられた反復可能オブジェクトで置き換えて self を返す。

        - 新しい ``list`` を生成して差し替えるため、元の ``_data`` 参照には影響しません。

        Args:
            iterable: 新しい要素列。

        Returns:
            Self: メソッドチェーン可能な自身。
        """
        self._data = list(iterable)
        return self

    # === Advanced Operations ===
    def filter(self, predicate: Callable[[T], bool]) -> Self:
        """述語を満たす要素のみからなる新しい同型コンテナを返す。"""
        return type(self)(item for item in self._data if predicate(item))

    def map(self, func: Callable[[T], Any]) -> "SequenceContainer[Any]":
        """各要素へ関数を適用した結果のコンテナを返す。"""
        return SequenceContainer(func(item) for item in self._data)

    def reduce(self, func: Callable[[Any, T], Any], initial: Any = None) -> Any:
        """関数で畳み込み単一値へ還元する。"""
        if initial is not None:
            result = initial
            for item in self._data:
                result = func(result, item)
            return result
        else:
            if not self._data:
                raise TypeError("reduce() of empty sequence with no initial value")
            result = self._data[0]
            for item in self._data[1:]:
                result = func(result, item)
            return result

    # === Split Operations ===
    def split_equal(self, n: int) -> list[Self]:
        """ほぼ等分となるよう n 分割する。

        Args:
            n: 分割数（正の整数）。

        Returns:
            同型の部分コンテナのリスト。

        Raises:
            ValueError: n が正でない場合。
        """
        if n <= 0:
            raise ValueError("Number of parts must be positive")

        # Use split_by_ratio with all equal ratios
        return self.split_by_ratio([1] * n)

    def split_by_ratio(self, ratios: list[float]) -> list[Self]:
        """与えた比率に従って分割する。

        Args:
            ratios: 各部分の比率（例: [1, 3, 4]）。

        Returns:
            同型の部分コンテナのリスト。

        Raises:
            ValueError: 比率に正でない値が含まれる場合。
        """

        if any(r <= 0 for r in ratios):
            raise ValueError("All ratios must be positive")

        if len(ratios) == 1:
            return [self.copy()]

        total_length = len(self._data)
        if total_length == 0:
            return [type(self)([]) for _ in range(len(ratios))]

        total_ratio = sum(ratios)
        result = []
        start = 0

        for i, ratio in enumerate(ratios):
            # For the last part, take all remaining elements to avoid rounding errors
            if i == len(ratios) - 1:
                end = total_length
            else:
                # Calculate size based on ratio, rounded to nearest integer
                size = round(total_length * ratio / total_ratio)
                end = min(start + size, total_length)

            result.append(type(self)(self._data[start:end]))
            start = end

        return result

    def split_by_size(self, size: int) -> list[Self]:
        """最大 `size` を上限として順に分割する。

        - 例: 要素数100・size=30 → [30, 30, 30, 10]

        Args:
            size: 各分割片の最大要素数（正の整数）。

        Returns:
            同型のコンテナのリスト。

        Raises:
            ValueError: `size` が正でない場合。
        """
        if size <= 0:
            raise ValueError("size must be positive")

        n = len(self._data)
        if n == 0:
            return []

        out: list[Self] = []
        for start in range(0, n, size):
            end = min(start + size, n)
            out.append(type(self)(self._data[start:end]))
        return out

    @classmethod
    def join(cls, containers: list[Self]) -> Self:
        """同型コンテナ列を結合して 1 つにまとめる。"""
        # 空のコンテナから開始
        result = cls([])

        # すべてのコンテナを順番に追加
        for container in containers:
            result = result + container
        return result
