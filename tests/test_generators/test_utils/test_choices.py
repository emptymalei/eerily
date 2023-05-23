from eerily.generators.utils.choices import Choices


def test_choices():
    elements = [0, 1]
    c = Choices(elements=elements)
    first_value = next(c)

    assert first_value in elements
