import pytest
from square import square


def test_square_positive_integer():
    """Test square function with positive integer."""
    assert square(3) == 9
    assert square(5) == 25


def test_square_negative_integer():
    """Test square function with negative integer."""
    assert square(-3) == 9
    assert square(-5) == 25


def test_square_zero():
    """Test square function with zero."""
    assert square(0) == 0


def test_square_float():
    """Test square function with floating point number."""
    assert square(2.5) == 6.25
    assert square(-1.5) == 2.25


def test_square_large_number():
    """Test square function with large number."""
    assert square(100) == 10000


def test_square_small_fraction():
    """Test square function with small fraction."""
    assert square(0.1) == pytest.approx(0.01)
    assert square(0.01) == pytest.approx(0.0001)