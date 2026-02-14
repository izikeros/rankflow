from rankflow.config import PlotConfig


def test_default_config():
    c = PlotConfig()
    assert c.line_width == 20
    assert c.title == "Rank evolution"
    assert c.caption == "Re-ranking step"
    assert c.show_metrics is False


def test_from_kwargs_ignores_unknown():
    c = PlotConfig.from_kwargs(line_width=10, unknown_field="ignored")
    assert c.line_width == 10


def test_override():
    c = PlotConfig()
    c2 = c.override(title="Custom Title", line_width=5)
    assert c2.title == "Custom Title"
    assert c2.line_width == 5
    # original unchanged
    assert c.title == "Rank evolution"
    assert c.line_width == 20


def test_to_dict():
    c = PlotConfig(title="Test")
    d = c.to_dict()
    assert isinstance(d, dict)
    assert d["title"] == "Test"


def test_instances_are_independent():
    c1 = PlotConfig()
    c2 = PlotConfig()
    c1.colors.append("extra_color")
    assert "extra_color" not in c2.colors
