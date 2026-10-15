from io import BytesIO

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

import patchworklib as pw


@pytest.fixture(autouse=True)
def clean_patchwork_state():
    pw.clear()
    yield
    pw.clear()


def make_brick(label):
    brick = pw.Brick(label=label, figsize=(2, 1.5))
    brick.plot([0, 1], [0, 1])
    return brick


@pytest.mark.parametrize("composite", [False, True])
@pytest.mark.parametrize("to_file", [False, True])
def test_quick_export_keeps_other_bricks_annotatable(composite, to_file):
    first, second, other = [make_brick(label) for label in ("a", "b", "c")]
    saved = first | second if composite else first
    saved.savefig(BytesIO() if to_file else None, format="png")

    annotation = other.text(0.1, 1.05, "Panel C", transform=other.transAxes)
    assert other.figure is not None
    assert annotation.figure is other.figure

    combined = other | saved
    rendered = combined.savefig(quick=False)
    rendered.canvas.draw()
    assert any(
        text.get_text() == "Panel C"
        for axis in rendered.axes
        for text in axis.texts
    )


@pytest.mark.parametrize("quick", [False, True])
@pytest.mark.parametrize("composite", [False, True])
def test_standalone_file_export_preserves_parent_layout(quick, composite):
    first, second, third = [make_brick(label) for label in ("a", "b", "c")]
    child = second / third if composite else second
    parent = first | child
    expected = {
        label: axis.get_position().bounds for label, axis in parent.bricks_dict.items()
    }

    child.savefig(BytesIO(), format="png", quick=quick)

    for label, axis in parent.bricks_dict.items():
        np.testing.assert_allclose(axis.get_position().bounds, expected[label])
    rendered = parent.savefig(quick=False)
    for axis in rendered.axes:
        if axis.get_label() in expected:
            np.testing.assert_allclose(
                axis.get_position().bounds, expected[axis.get_label()]
            )


def test_isolated_figure_export_preserves_live_layout():
    first, second = make_brick("a"), make_brick("b")
    parent = first | second
    expected = second.get_position().bounds

    rendered = second.savefig(quick=False)

    np.testing.assert_allclose(second.get_position().bounds, expected)
    assert rendered is not pw.basefigure
    assert {axis.get_label() for axis in rendered.axes} == {"b", "case:b"}
    data_axis = next(axis for axis in rendered.axes if axis.get_label() == "b")
    np.testing.assert_allclose(data_axis.get_position().bounds, (0, 0, 2, 1.5))
    assert set(parent.bricks_dict) == {"a", "b"}


def test_failed_export_preserves_layout_and_other_bricks(tmp_path):
    first, second, other = [make_brick(label) for label in ("a", "b", "c")]
    parent = first | second
    expected = second.get_position().bounds

    with pytest.raises(FileNotFoundError):
        second.savefig(tmp_path / "missing" / "figure.png")

    np.testing.assert_allclose(second.get_position().bounds, expected)
    annotation = other.text(0.1, 1.05, "Still usable", transform=other.transAxes)
    assert annotation.figure is other.figure
    (parent | other).savefig(BytesIO(), format="png")


@pytest.mark.parametrize("quick", [False, True])
@pytest.mark.parametrize("polar", [False, True])
def test_export_preserves_fixed_aspect_axes_positions(quick, polar):
    if polar:
        brick = pw.cBrick(label="polar", figsize=(3, 2))
        brick.plot([0, 1, 2], [1, 2, 1])
    else:
        brick = pw.Brick(label="image", figsize=(3, 2))
        brick.imshow([[0, 1], [1, 0]])
    original = brick.get_position(original=True).bounds
    active = brick.get_position().bounds

    brick.savefig(BytesIO(), format="png", quick=quick)

    np.testing.assert_allclose(brick.get_position(original=True).bounds, original)
    np.testing.assert_allclose(brick.get_position().bounds, active)


def test_repeated_nested_exports_keep_axes_and_page_bounds():
    first, second, third = [make_brick(label) for label in ("a", "b", "c")]
    group = first / second
    group.set_supxlabel("Shared x")
    group.set_supylabel("Shared y")
    group.set_suptitle("Shared title")
    figure = group | third
    figure.change_plotsize((7.2, 4.0))

    snapshots = []
    for _ in range(3):
        rendered = figure.savefig(quick=False)
        rendered.canvas.draw()
        snapshots.append(
            (
                rendered.get_tightbbox(rendered.canvas.get_renderer()).bounds,
                {
                    axis.get_label(): axis.get_position().bounds
                    for axis in rendered.axes
                },
            )
        )

    for bounds, positions in snapshots[1:]:
        np.testing.assert_allclose(bounds, snapshots[0][0])
        assert set(positions) == set(snapshots[0][1])
        for label, position in positions.items():
            np.testing.assert_allclose(position, snapshots[0][1][label])


def test_polar_panel_composes_with_cartesian_panel():
    polar = pw.cBrick(label="polar", figsize=(2, 2))
    polar.plot([0, 1, 2], [1, 2, 1])
    cartesian = make_brick("cartesian")

    rendered = (polar | cartesian).savefig(quick=False)
    rendered.canvas.draw()

    assert {axis.get_label() for axis in rendered.axes} >= {"polar", "cartesian"}
    polar_axis = next(axis for axis in rendered.axes if axis.get_label() == "polar")
    assert polar_axis.name == "polar"
