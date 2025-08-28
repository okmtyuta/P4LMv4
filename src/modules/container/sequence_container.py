from __future__ import annotations

from typing import Any, Callable, Iterable, Iterator, Optional, Self, overload


class SequenceContainer[T]:
    """A generic sequence container that mimics list behavior with type safety."""

    def __init__(self, iterable: Iterable[T]) -> None:
        self._data = list(iterable)

    # === Basic Properties ===
    def __len__(self) -> int:
        """Return the number of elements in the container."""
        return len(self._data)

    def __bool__(self) -> bool:
        """Return True if container is not empty."""
        return bool(self._data)

    @property
    def is_empty(self) -> bool:
        """Return True if container is empty."""
        return len(self._data) == 0

    # === Iteration ===
    def __iter__(self) -> Iterator[T]:
        """Return an iterator over the elements."""
        return iter(self._data)

    def __reversed__(self) -> Iterator[T]:
        """Return a reverse iterator over the elements."""
        return reversed(self._data)

    # === Membership ===
    def __contains__(self, item: T) -> bool:
        """Return True if item is in the container."""
        return item in self._data

    # === Item Access ===
    @overload
    def __getitem__(self, i: int) -> T:
        pass

    @overload
    def __getitem__(self, s: slice) -> Self:
        pass

    def __getitem__(self, key: int | slice) -> T | Self:
        """Get item(s) at the specified index or slice."""
        if isinstance(key, slice):
            return type(self)(self._data[key])
        return self._data[key]

    def __setitem__(self, key: int | slice, value: T | Iterable[T]) -> None:
        """Set item(s) at the specified index or slice."""
        if isinstance(key, slice):
            if not isinstance(value, Iterable):
                raise TypeError("can only assign an iterable to a slice")
            self._data[key] = list(value)
        else:
            # For single index assignment, value should be T, not Iterable[T]
            # We'll cast to T to satisfy type checker
            self._data[key] = value  # type: ignore[assignment]

    def __delitem__(self, key: int | slice) -> None:
        """Delete item(s) at the specified index or slice."""
        del self._data[key]

    # === Arithmetic Operations ===
    def __add__(self, other: Iterable[T]) -> Self:
        """Return a new container with elements from this container and other."""
        return type(self)(self._data + list(other))

    def __radd__(self, other: Iterable[T]) -> Self:
        """Return a new container with elements from other and this container."""
        return type(self)(list(other) + self._data)

    def __iadd__(self, other: Iterable[T]) -> Self:
        """Extend this container with elements from other."""
        self.extend(other)
        return self

    def __mul__(self, n: int) -> Self:
        """Return a new container with elements repeated n times."""
        return type(self)(self._data * n)

    def __rmul__(self, n: int) -> Self:
        """Return a new container with elements repeated n times."""
        return type(self)(n * self._data)

    def __imul__(self, n: int) -> Self:
        """Repeat elements in this container n times."""
        self._data *= n
        return self

    # === String Representation ===
    def __str__(self) -> str:
        """Return a string representation of the container."""
        return f"SequenceContainer({self._data})"

    def __repr__(self) -> str:
        """Return a detailed string representation of the container."""
        return f"SequenceContainer({self._data!r})"

    # === Modification Operations ===
    def append(self, item: T) -> None:
        """Add an item to the end of the container."""
        self._data.append(item)

    def insert(self, index: int, item: T) -> None:
        """Insert an item at the specified index."""
        self._data.insert(index, item)

    def remove(self, item: T) -> None:
        """Remove the first occurrence of item from the container."""
        self._data.remove(item)

    def pop(self, index: int = -1) -> T:
        """Remove and return the item at the specified index (default: last)."""
        return self._data.pop(index)

    def clear(self) -> None:
        """Remove all items from the container."""
        self._data.clear()

    def extend(self, iterable: Iterable[T]) -> None:
        """Add all items from iterable to the end of the container."""
        self._data.extend(iterable)

    def reverse(self) -> None:
        """Reverse the elements in the container in place."""
        self._data.reverse()

    def sort(self, *, key: Optional[Callable[[T], Any]] = None, reverse: bool = False) -> None:
        """Sort the elements in the container in place."""
        self._data.sort(key=key, reverse=reverse)

    # === Search and Count Operations ===
    def index(self, item: T, start: int = 0, stop: Optional[int] = None) -> int:
        """Return the index of the first occurrence of item."""
        if stop is None:
            return self._data.index(item, start)
        return self._data.index(item, start, stop)

    def count(self, item: T) -> int:
        """Return the number of occurrences of item in the container."""
        return self._data.count(item)

    # === Utility Methods ===
    def copy(self) -> Self:
        """Return a shallow copy of the container."""
        return type(self)(self._data)

    def to_list(self) -> list[T]:
        """Return a copy of the internal list."""
        return self._data.copy()

    # === Advanced Operations ===
    def filter(self, predicate: Callable[[T], bool]) -> Self:
        """Return a new container with elements that satisfy the predicate."""
        return type(self)(item for item in self._data if predicate(item))

    def map(self, func: Callable[[T], Any]) -> "SequenceContainer[Any]":
        """Return a new container with function applied to each element."""
        return SequenceContainer(func(item) for item in self._data)

    def reduce(self, func: Callable[[Any, T], Any], initial: Any = None) -> Any:
        """Reduce the container to a single value using the function."""
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
        """Split container into n equal parts (or as equal as possible).

        Args:
            n: Number of parts to split into.

        Returns:
            List of containers of the same type as self.

        Raises:
            ValueError: If n <= 0.
        """
        if n <= 0:
            raise ValueError("Number of parts must be positive")

        # Use split_by_ratio with all equal ratios
        return self.split_by_ratio([1] * n)

    def split_by_ratio(self, ratios: list[float]) -> list[Self]:
        """Split container by the given ratios.

        Args:
            ratios: List of integers representing the ratio for each part.
                   e.g., [1, 3, 4] means split in ratio 1:3:4

        Returns:
            List of containers of the same type as self.

        Raises:
            ValueError: If ratios contains non-positive values.
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

    @classmethod
    def join(cls, containers: list[Self]) -> Self:
        """Join multiple containers of the same type into one.

        Args:
            containers: List of containers to join.

        Returns:
            A new container of the same type containing all elements.
        """
        # 空のコンテナから開始
        result = cls([])

        # すべてのコンテナを順番に追加
        for container in containers:
            result = result + container
        return result
