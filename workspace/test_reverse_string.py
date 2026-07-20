import pytest

from reverse_string import reverse_string


def test_reverse_empty_string():
    """Test reversing an empty string."""
    assert reverse_string("") == ""


def test_reverse_single_character():
    """Test reversing a single character string."""
    assert reverse_string("a") == "a"


def test_reverse_multiple_characters():
    """Test reversing a string with multiple characters."""
    assert reverse_string("hello") == "olleh"
    assert reverse_string("world") == "dlrow"
    assert reverse_string("python") == "nohtyp"


def test_reverse_unicode_string():
    """Test reversing a Unicode string."""
    assert reverse_string("café") == "éfac"
    assert reverse_string("🌍🚀") == "🚀🌍"


def test_reverse_numbers_as_string():
    """Test reversing a string containing numbers."""
    assert reverse_string("12345") == "54321"
    assert reverse_string("abc123") == "321cba"


def test_reverse_with_spaces_and_punctuation():
    """Test reversing a string with spaces and punctuation."""
    assert reverse_string("hello world!") == "!dlrow olleh"
    assert reverse_string("a.b,c!") == "!c,b.a"


def test_type_error_for_non_string_input():
    """Test that TypeError is raised for non-string inputs."""
    with pytest.raises(TypeError) as exc_info:
        reverse_string(123)
    assert "Input must be a string" in str(exc_info.value)
    
    with pytest.raises(TypeError) as exc_info:
        reverse_string([1, 2, 3])
    assert "Input must be a string" in str(exc_info.value)
    
    with pytest.raises(TypeError) as exc_info:
        reverse_string(None)
    assert "Input must be a string" in str(exc_info.value)


def test_reverse_special_cases():
    """Test reversing special cases like whitespace-only strings."""
    assert reverse_string(" ") == " "
    assert reverse_string("  ") == "  "
    assert reverse_string("\t\n\r") == "\r\n\t"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
