from __future__ import annotations

import dataclasses
import importlib
import inspect
from typing import Any, ClassVar, Dict, Self, get_args, get_origin, get_type_hints


class SerializableObject:
    """Base class for objects that can be converted to/from dictionary representation.

    Provides serialization and deserialization capabilities for Python objects,
    supporting nested structures, type hints, and both dataclasses and regular classes.
    """

    # Keys for nested object markers
    instance_key: ClassVar[str] = "__instance__"  # accepted on input
    nested_flag_key: ClassVar[str] = "__dc__"
    nested_type_key: ClassVar[str] = "__type__"
    nested_data_key: ClassVar[str] = "__data__"

    # --- internal helpers ---
    @classmethod
    def _instance_field_names(cls) -> set[str]:
        """Extract instance field names from type hints, excluding ClassVar fields.

        Returns:
            Set of field names that are instance attributes (not class variables).
        """
        hints = get_type_hints(cls, include_extras=True)
        return {k for k, t in hints.items() if get_origin(t) is not ClassVar}

    def _instance_dict(self) -> Dict[str, Any]:
        """Extract instance attributes as a dictionary.

        Uses type hints when available, falls back to __dict__ or __slots__.
        Class variables are intentionally ignored.

        Returns:
            Dictionary mapping attribute names to their values.
        """
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
        """Convert a type to a string representation for serialization.

        Args:
            tp: The type to convert.

        Returns:
            String in format 'module_name:ClassName'.
        """
        return f"{tp.__module__}:{tp.__qualname__}"

    @classmethod
    def _type_from_string(cls, s: str) -> type:
        """Convert a string representation back to a type.

        Args:
            s: String in format 'module_name:ClassName'.

        Returns:
            The reconstructed type object.
        """
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
        """Recursively serialize a value, handling nested SerializableObject objects.

        Args:
            value: The value to serialize.
            include_class_vars: Whether to include class variables (currently ignored).
            annotated_class_only: Whether to only process annotated classes.

        Returns:
            Serialized representation of the value.
        """
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
        """Decode a value without type hints, using nested object markers.

        Recursively processes containers and reconstructs SerializableObject objects
        based on nested flags and type information.

        Args:
            value: The value to decode.

        Returns:
            Decoded Python object.
        """
        # Wrapper produced by _deep_serialize
        if (
            isinstance(value, dict)
            and value.get(cls.nested_flag_key) is True
            and cls.nested_type_key in value
            and cls.nested_data_key in value
        ):
            tp = cls._type_from_string(value[cls.nested_type_key])
            if issubclass(tp, SerializableObject):
                return tp.from_dict(value[cls.nested_data_key])
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
        """Decode a value using type hints for better type safety.

        Uses type hints to guide the deserialization process, handling
        generic types like List[T], Dict[K, V], and direct SerializableObject types.

        Args:
            value: The value to decode.
            hint: Type hint to guide decoding.

        Returns:
            Decoded object with appropriate type.
        """
        if hint is None:
            return cls._decode_untyped(value)

        origin = get_origin(hint)
        args = get_args(hint)

        # Direct SerializableObject type
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
        """Convert this object to a dictionary representation.

        Serializes the object's instance attributes to a dictionary,
        recursively handling nested SerializableObject objects.

        Args:
            include_class_vars: Ignored, kept for compatibility.
            annotated_class_only: Ignored for class vars.

        Returns:
            Dictionary representation of this object.
        """
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
        """Create an instance of this class from a dictionary representation.

        Deserializes a dictionary back into an object instance, handling
        nested objects and type hints appropriately.

        Args:
            data: Dictionary containing the object data.
            set_class_vars: Ignored, kept for compatibility.
            annotated_class_only: Whether to only process annotated classes.
            ignore_unknown: Whether to ignore unknown fields.

        Returns:
            New instance of this class initialized with the dictionary data.
        """
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
