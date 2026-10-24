import json
import logging

from repomind.logging_config import JsonFormatter


def test_json_formatter_emits_one_json_object_with_extras() -> None:
    record = logging.LogRecord(
        "repomind.test", logging.INFO, __file__, 1, "hello %s", ("world",), None
    )
    record.request_id = "r1"

    payload = json.loads(JsonFormatter().format(record))

    assert payload["message"] == "hello world"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "repomind.test"
    assert payload["request_id"] == "r1"
    assert "timestamp" in payload
