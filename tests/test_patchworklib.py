from pathlib import Path

import pandas as pd
import pytest
import seaborn as sns

import patchworklib as pw


@pytest.fixture(autouse=True)
def clean_patchwork_state():
    pw.clear()
    yield
    pw.clear()


def test_example_plot(tmp_path: Path):
    """Test example plot"""
    fmri = pd.DataFrame({
        "timepoint": [0, 1, 2, 0, 1, 2],
        "signal": [0.1, 0.3, 0.2, 0.2, 0.4, 0.3],
        "region": ["a", "a", "a", "b", "b", "b"],
        "event": ["stimulus"] * 6,
    })
    ax1 = pw.Brick(figsize=(3, 2))
    sns.lineplot(
        x="timepoint", y="signal", hue="region", style="event", data=fmri, ax=ax1
    )
    ax1.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
    ax1.set_title("ax1")

    titanic = pd.DataFrame({
        "sex": ["female", "male", "female", "male"],
        "survived": [1, 0, 1, 1],
        "class": ["First", "First", "Third", "Third"],
    })
    ax2 = pw.Brick(figsize=(1, 2))
    sns.barplot(x="sex", y="survived", hue="class", data=titanic, ax=ax2)
    ax2.move_legend(new_loc="upper left", bbox_to_anchor=(1.05, 1.0))
    ax2.set_title("ax2")

    result_file = tmp_path / "ax12.png"
    ax12 = ax1 | ax2
    ax12.savefig(result_file)
    assert result_file.exists()


def test_seaborn_and_matplotlib(tmp_path: Path):
    titanic = pd.DataFrame({
        "sex": ["female", "male", "female", "male"],
        "survived": [1, 0, 1, 1],
        "class": ["First", "First", "Third", "Third"],
    })

    g_sns = pw.Brick(figsize=(4, 4))
    sns.boxplot(data=titanic, x="sex", y="survived", hue="class", ax=g_sns)
    g_sns.set_title("seaborn")

    g_mpl = pw.Brick(figsize=(4, 4))
    g_mpl.plot([0, 1, 2], [0, 1, 0])
    g_mpl.set_title("matplotlib")

    g = g_sns | g_mpl

    result_file = tmp_path / "g.png"
    g.savefig(result_file)
    assert result_file.exists()
