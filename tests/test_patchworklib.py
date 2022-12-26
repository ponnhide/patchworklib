from pathlib import Path

import seaborn as sns

import patchworklib as pw


def test_example_plot(tmp_path: Path):
    """Test example plot"""
    fmri = sns.load_dataset("fmri")
    ax1 = pw.Brick(figsize=(3, 2))
    sns.lineplot(
        x="timepoint", y="signal", hue="region", style="event", data=fmri, ax=ax1
    )
    ax1.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
    ax1.set_title("ax1")

    titanic = sns.load_dataset("titanic")
    ax2 = pw.Brick(figsize=(1, 2))
    sns.barplot(x="sex", y="survived", hue="class", data=titanic, ax=ax2)
    ax2.move_legend(new_loc="upper left", bbox_to_anchor=(1.05, 1.0))
    ax2.set_title("ax2")

    result_file = tmp_path / "ax12.png"
    ax12 = ax1 | ax2
    ax12.savefig(result_file)
    assert result_file.exists()
