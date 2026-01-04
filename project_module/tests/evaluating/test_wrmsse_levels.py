from __future__ import annotations

from project_module.evaluating.wrmsse import get_level_specs


def test_get_level_specs_includes_state_category() -> None:
    specs = get_level_specs()
    names = [spec.name for spec in specs]
    assert len(specs) == 12
    assert "state_category" in names
    state_category = next(spec for spec in specs if spec.name == "state_category")
    assert state_category.series_count == 9
    assert state_category.group_cols == ["state_id", "cat_id"]
