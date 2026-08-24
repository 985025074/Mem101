from memkernel.provenance import RegexSourceSanitizer


def test_sanitizer_redacts_common_secrets() -> None:
    sanitizer = RegexSourceSanitizer()
    source = (
        "password=hunter2 "
        "Authorization: Bearer abcdefghijklmnop "
        "token sk-abcdefghijklmnop"
    )

    sanitized = sanitizer.sanitize_text(source)

    assert "hunter2" not in sanitized
    assert "abcdefghijklmnop" not in sanitized
    assert "password=[REDACTED:SECRET]" in sanitized
    assert "Bearer [REDACTED:TOKEN]" in sanitized


def test_sanitizer_redacts_nested_metadata() -> None:
    sanitizer = RegexSourceSanitizer()

    sanitized = sanitizer.sanitize_metadata(
        {
            "safe": "visible",
            "auth-token": "secret-value",
            "nested": {"password": "hidden"},
        }
    )

    assert sanitized == {
        "safe": "visible",
        "auth-token": "[REDACTED:SECRET]",
        "nested": {"password": "[REDACTED:SECRET]"},
    }
