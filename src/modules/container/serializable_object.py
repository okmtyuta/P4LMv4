from __future__ import annotations

"""
辞書との相互変換（シリアライズ/デシリアライズ）を提供する基底クラス。

- ネストしたオブジェクトや型ヒント、dataclass/通常クラスの双方を扱います。
"""

import dataclasses
import importlib
import inspect
from typing import Any, ClassVar, Dict, Self, get_args, get_origin, get_type_hints

from src.modules.container.sequence_container import SequenceContainer


class SerializableObject:
    """辞書への変換と、辞書からの復元を提供する基底クラス。"""

    # Keys for nested object markers
    instance_key: ClassVar[str] = "__instance__"  # accepted on input
    nested_flag_key: ClassVar[str] = "__dc__"
    nested_type_key: ClassVar[str] = "__type__"
    nested_data_key: ClassVar[str] = "__data__"

    # --- internal helpers ---
    @classmethod
    def _instance_field_names(cls) -> set[str]:
        """型ヒントからインスタンス属性の名前集合を抽出する（ClassVar除外）。"""
        hints = get_type_hints(cls, include_extras=True)
        return {k for k, t in hints.items() if get_origin(t) is not ClassVar}

    def _instance_dict(self) -> Dict[str, Any]:
        """インスタンス属性を辞書として取得する（型ヒント/`__dict__`/`__slots__` を使用）。"""
        # Use type hints when available (only attributes that exist)
        names = type(self)._instance_field_names()
        if names:
            return {k: getattr(self, k) for k in names if hasattr(self, k)}
        # Fallbacks: __dict__ or __slots__
        if hasattr(self, "__dict__"):
            return dict(self.__dict__)
        if hasattr(self, "__slots__"):
            out: Dict[str, Any] = {}
            for s in self.__slots__:  # type: ignore[attr-defined]
                if hasattr(self, s):
                    out[s] = getattr(self, s)
            return out
        return {}

    # Note: Class variables are intentionally ignored in serialization.

    # --- public API ---
    @classmethod
    def _type_to_string(cls, tp: type) -> str:
        """型を 'module:Class' 形式の文字列へ変換する。"""
        return f"{tp.__module__}:{tp.__qualname__}"

    @classmethod
    def _type_from_string(cls, s: str) -> type:
        """'module:Class' 形式の文字列から型を復元する。"""
        module, qual = s.split(":", 1)
        mod = importlib.import_module(module)
        obj: Any = mod
        for part in qual.split("."):
            obj = getattr(obj, part)
        return obj  # type: ignore[return-value]

    @classmethod
    def _deep_serialize(
        cls,
        value: Any,
        *,
        include_class_vars: bool,
        annotated_class_only: bool,
    ) -> Any:
        """値を再帰的にシリアライズする（SerializableObject/コンテナに対応）。"""
        # SerializableObject instance
        if isinstance(value, SerializableObject):
            return {
                cls.nested_flag_key: True,
                cls.nested_type_key: cls._type_to_string(type(value)),
                cls.nested_data_key: value.to_dict(
                    include_class_vars=include_class_vars,
                    annotated_class_only=annotated_class_only,
                ),
            }
        # SequenceContainer (e.g., SerializableContainerList): wrap with type + data for round-trip
        if isinstance(value, SequenceContainer):
            return {
                cls.nested_flag_key: True,
                cls.nested_type_key: cls._type_to_string(type(value)),
                cls.nested_data_key: [
                    cls._deep_serialize(
                        v,
                        include_class_vars=include_class_vars,
                        annotated_class_only=annotated_class_only,
                    )
                    for v in value
                ],
            }

        # Containers
        if isinstance(value, dict):
            return {
                k: cls._deep_serialize(
                    v,
                    include_class_vars=include_class_vars,
                    annotated_class_only=annotated_class_only,
                )
                for k, v in value.items()
            }
        if isinstance(value, list):
            return [
                cls._deep_serialize(
                    v,
                    include_class_vars=include_class_vars,
                    annotated_class_only=annotated_class_only,
                )
                for v in value
            ]
        if isinstance(value, tuple):
            return tuple(
                cls._deep_serialize(
                    v,
                    include_class_vars=include_class_vars,
                    annotated_class_only=annotated_class_only,
                )
                for v in value
            )
        # Primitives / others
        return value

    @classmethod
    def _decode_untyped(cls, value: Any) -> Any:
        """型ヒントなしでネスト情報を頼りに復号する。"""
        # Wrapper produced by _deep_serialize
        if (
            isinstance(value, dict)
            and value.get(cls.nested_flag_key) is True
            and cls.nested_type_key in value
            and cls.nested_data_key in value
        ):
            tp = cls._type_from_string(value[cls.nested_type_key])
            # SerializableObject -> use from_dict
            if issubclass(tp, SerializableObject):
                return tp.from_dict(value[cls.nested_data_key])
            # SequenceContainer -> construct from decoded sequence
            if issubclass(tp, SequenceContainer):
                seq_raw = value[cls.nested_data_key]
                if not isinstance(seq_raw, (list, tuple)):
                    return seq_raw
                seq = [cls._decode_untyped(v) for v in seq_raw]
                return tp(seq)
        # Recurse containers
        if isinstance(value, dict):
            return {k: cls._decode_untyped(v) for k, v in value.items()}
        if isinstance(value, list):
            return [cls._decode_untyped(v) for v in value]
        if isinstance(value, tuple):
            return tuple(cls._decode_untyped(v) for v in value)
        return value

    @classmethod
    def _decode_by_hint(cls, value: Any, hint: Any) -> Any:
        """型ヒントを用いてより厳密に復号する（リスト/辞書/オブジェクト対応）。"""
        if hint is None:
            return cls._decode_untyped(value)

        origin = get_origin(hint)
        args = get_args(hint)

        # Direct SerializableObject type (supports generic alias origin=None)
        try:
            if origin is None and isinstance(hint, type) and issubclass(hint, SerializableObject):
                if isinstance(value, dict) and (value.get(cls.nested_flag_key) is True or cls.instance_key in value):
                    return hint.from_dict(value.get(cls.nested_data_key, value))
                # Even without wrapper markers, if a SerializableObject is expected,
                # treat plain dicts as constructor data for that type.
                if isinstance(value, dict):
                    return hint.from_dict(value)
        except Exception:
            pass

        # SequenceContainer[T]-like types (either direct class or generic alias origin)
        try:
            container_cls = None
            elem_hint = None
            if origin is not None and isinstance(origin, type) and issubclass(origin, SequenceContainer):
                container_cls = origin
                elem_hint = args[0] if args else None
            elif origin is None and isinstance(hint, type) and issubclass(hint, SequenceContainer):
                container_cls = hint
                elem_hint = None
            if container_cls is not None:
                # Wrapper case
                if isinstance(value, dict) and value.get(cls.nested_flag_key) is True and cls.nested_data_key in value:
                    seq_raw = value[cls.nested_data_key]
                    if hasattr(seq_raw, "tolist"):
                        seq_raw = seq_raw.tolist()
                    seq = [cls._decode_by_hint(v, elem_hint) for v in (seq_raw or [])]
                    return container_cls(seq)
                # Plain list case
                if hasattr(value, "tolist"):
                    value = value.tolist()
                if isinstance(value, list):
                    seq = [cls._decode_by_hint(v, elem_hint) for v in value]
                    return container_cls(seq)
        except Exception:
            pass

        # Containers with type args
        if origin in (list, tuple):
            elem_hint = args[0] if args else None
            # Handle numpy arrays by converting to list first
            if hasattr(value, "tolist"):
                value = value.tolist()
            seq = [cls._decode_by_hint(v, elem_hint) for v in (value or [])]
            return tuple(seq) if origin is tuple else seq
        if origin is dict:
            key_hint = args[0] if len(args) > 0 else None
            val_hint = args[1] if len(args) > 1 else None
            if isinstance(value, dict):
                return {cls._decode_by_hint(k, key_hint): cls._decode_by_hint(v, val_hint) for k, v in value.items()}

        # Fallback to untyped recursive decode
        return cls._decode_untyped(value)

    def to_dict(
        self,
        *,
        include_class_vars: bool = False,  # ignored; kept for compatibility
        annotated_class_only: bool = True,  # ignored for class vars
    ) -> Dict[str, Any]:
        """このオブジェクトを辞書に変換する（入れ子も再帰的に処理）。"""
        inst_raw = self._instance_dict()
        inst = {
            k: type(self)._deep_serialize(
                v,
                include_class_vars=False,
                annotated_class_only=annotated_class_only,
            )
            for k, v in inst_raw.items()
        }
        # Return only instance fields
        return dict(inst)

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        *,
        set_class_vars: bool = True,
        annotated_class_only: bool = True,
        ignore_unknown: bool = False,
    ) -> Self:
        """辞書からインスタンスを復元する（型ヒント/入れ子に対応）。"""
        # 1) Determine dict layout (prefer nested instance key). Class vars ignored.
        if cls.instance_key in data:
            inst_part = data.get(cls.instance_key, {}) or {}
        else:
            # Flat dict: treat all keys as instance fields
            inst_part = dict(data)

        # 3) Construct instance
        if dataclasses.is_dataclass(cls):
            fields = {f.name for f in dataclasses.fields(cls) if f.init}
            hints = get_type_hints(cls, include_extras=True)
            inst_hints = {k: t for k, t in hints.items() if get_origin(t) is not ClassVar}
            kwargs = {k: cls._decode_by_hint(v, inst_hints.get(k)) for k, v in inst_part.items() if k in fields}
            return cls(**kwargs)  # type: ignore[misc]

        # Non-dataclass: inspect __init__ and pass acceptable kwargs
        try:
            sig = inspect.signature(cls)
            params = {
                k
                for k, p in sig.parameters.items()
                if k != "self" and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
            }
            hints = get_type_hints(cls, include_extras=True)
            inst_hints = {k: t for k, t in hints.items() if get_origin(t) is not ClassVar}
            kwargs = {k: cls._decode_by_hint(v, inst_hints.get(k)) for k, v in inst_part.items() if k in params}
            obj = cls(**kwargs)  # type: ignore[misc]
        except Exception:
            # Fallback: no-arg construct then assign attributes
            obj = cls()  # type: ignore[call-arg]
            hints = get_type_hints(cls, include_extras=True)
            inst_hints = {k: t for k, t in hints.items() if get_origin(t) is not ClassVar}
            for k, v in inst_part.items():
                setattr(obj, k, cls._decode_by_hint(v, inst_hints.get(k)))
        return obj
