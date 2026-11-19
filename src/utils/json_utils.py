import json
from typing import Any, Dict


class StructuredResponseError(RuntimeError):
    pass
def extract_json_object(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise StructuredResponseError("The model returned an empty response")
    cleaned = text.strip()
    if "```json" in cleaned:
        cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in cleaned:
        cleaned = cleaned.split("```", 1)[1].split("```", 1)[0].strip()

    start = cleaned.find("{")
    if start < 0:
        return cleaned
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(cleaned)):
        character = cleaned[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return cleaned[start:index + 1]
    return cleaned[start:]


def parse_json_object(text: str) -> Dict[str, Any]:
    candidate = extract_json_object(text)
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise StructuredResponseError("The model did not return valid JSON: %s" % exc) from exc
    if not isinstance(value, dict):
        raise StructuredResponseError("The model JSON response must be an object")
    return value
