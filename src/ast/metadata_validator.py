from typing import Any, Dict, Union, List, Optional


class MetadataValidator:
    def __init__(self, metadata_schema: Dict[str, Any]):
        """
        metadata_schema must contain:
          - "global"  : dict
          - "class"   : dict
          - "function": dict
        """
        self.global_metadata_schema: Dict[str, Any] = metadata_schema['global']
        self.class_metadata_schema: Dict[str, Any] = metadata_schema['class']
        self.function_metadata_schema: Dict[str, Any] = metadata_schema['function']

        # quick ref map for resolving "$ref"
        self._ref_map = {
            "class_metadata_schema": self.class_metadata_schema,
            "function_metadata_schema": self.function_metadata_schema,
            "global_metadata_schema": self.global_metadata_schema,
        }

    # ---------- public API ----------
    def validate(self, data: Dict[str, Any]) -> List[str]:
        """
        Validate top-level `data` (global metadata), then drill into classes and functions.
        Return list of errors. Empty list means valid.
        """
        errors: List[str] = []

        if not isinstance(data, dict):
            return [f"Top-level metadata must be an object/dict, got {type(data).__name__}"]

        # 1. Global
        errors += self._validate_object(data, self.global_metadata_schema, path="global")

        # 2. Classes + methods
        if data.get("classes"):
            for i, cls in enumerate(data["classes"]):
                errors += self._validate_object(cls, self.class_metadata_schema, path=f"class[{i}]")
                if cls.get("methods"):
                    for j, m in enumerate(cls["methods"]):
                        errors += self._validate_object(m, self.function_metadata_schema, path=f"class[{i}].method[{j}]")

        # 3. Top-level functions
        if data.get("functions"):
            for i, fn in enumerate(data["functions"]):
                errors += self._validate_object(fn, self.function_metadata_schema, path=f"function[{i}]")

        # 4. Rule: require at least one class or function
        if not self._is_nonempty(data.get("classes")) and not self._is_nonempty(data.get("functions")):
            errors.append("At least one of 'classes' or 'functions' must be a non-empty list.")

        return errors

    def validate_or_raise(self, data: Dict[str, Any]) -> None:
        errs = self.validate(data)
        if errs:
            raise ValueError("Metadata validation failed:\n" + "\n".join(errs))

    # ---------- internal helpers ----------
    def _validate_object(self, obj: Dict[str, Any], schema: Dict[str, Any], path: str) -> List[str]:
        errs: List[str] = []
        for prop, rule in schema.items():
            prop_path = f"{path}.{prop}"
            if prop not in obj:
                errs.append(f"Missing key: {prop_path}")
                continue

            val = obj[prop]
            expected_type = rule.get("type")
            allowed_values = rule.get("value", "any")

            # type check
            if not self._check_type(val, expected_type):
                errs.append(f"Key '{prop_path}' has invalid type. Expected {expected_type}, got {type(val).__name__}")
                continue

            # value check
            if allowed_values != "any" and val is not None:
                if isinstance(allowed_values, list) and val not in allowed_values:
                    errs.append(f"Key '{prop_path}' has invalid value {val}. Allowed: {allowed_values}")
                elif isinstance(allowed_values, (str, int, bool)) and val != allowed_values:
                    errs.append(f"Key '{prop_path}' has invalid value {val}. Expected: {allowed_values}")

            # array items
            if self._is_type(expected_type, "array") and val is not None:
                items_rule = rule.get("items")
                if items_rule:
                    item_schema = self._resolve_ref_or_schema(items_rule)
                    if isinstance(item_schema, dict) and self._looks_like_schema(item_schema):
                        for i, elem in enumerate(val):
                            if not isinstance(elem, dict):
                                errs.append(f"{prop_path}[{i}] expected object/dict, got {type(elem).__name__}")
                                continue
                            errs += self._validate_object(elem, item_schema, path=f"{prop_path}[{i}]")
                    elif isinstance(item_schema, dict) and "type" in item_schema:
                        for i, elem in enumerate(val):
                            if not self._check_type(elem, item_schema["type"]):
                                errs.append(f"{prop_path}[{i}] has invalid type. Expected {item_schema['type']}, got {type(elem).__name__}")
                            else:
                                allowed = item_schema.get("value", "any")
                                if allowed != "any" and elem not in allowed:
                                    errs.append(f"{prop_path}[{i}] invalid value {elem}. Allowed {allowed}")
        return errs

    @staticmethod
    def _is_nonempty(v) -> bool:
        return isinstance(v, list) and len(v) > 0

    def _resolve_ref_or_schema(self, node: Any) -> Any:
        if isinstance(node, dict) and "$ref" in node:
            ref_name = node["$ref"]
            return self._ref_map.get(ref_name, node)
        return node

    def _looks_like_schema(self, node: Any) -> bool:
        if not isinstance(node, dict):
            return False
        return any(isinstance(v, dict) and "type" in v for v in node.values())

    def _is_type(self, expected_type: Union[str, List[str], None], check_type: str) -> bool:
        if expected_type is None:
            return False
        if isinstance(expected_type, str):
            return expected_type == check_type
        if isinstance(expected_type, list):
            return check_type in expected_type
        return False

    def _check_type(self, value: Any, expected_type: Union[str, List[str], None]) -> bool:
        type_map = {
            "string": str,
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict,
            "null": type(None),
        }
        if expected_type is None:
            return True
        if isinstance(expected_type, str):
            pytype = type_map.get(expected_type)
            return isinstance(value, pytype) if pytype else True
        if isinstance(expected_type, list):
            return any(isinstance(value, type_map.get(t, object)) for t in expected_type)
        return True
