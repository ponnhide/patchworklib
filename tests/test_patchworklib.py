from pathlib import Path

import plotnine as p9
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


def test_sns_and_p9(tmp_path: Path):
    titanic = sns.load_dataset("titanic")

    g_sns = pw.Brick(figsize=(4, 4))
    sns.boxplot(data=titanic, x="sex", y="survived", hue="class", ax=g_sns)
    g_sns.set_title("seaborn")

    g_p9 = pw.load_ggplot(
        (
            p9.ggplot(titanic, p9.aes(x="sex", y="survived", fill="class"))
            + p9.geom_boxplot()
            + p9.ggtitle("plotnine")
        ),
        figsize=(4, 4),
    )

    g = g_sns | g_p9

    result_file = tmp_path / "g.png"
    g.savefig(result_file)
    assert result_file.exists()


@pw.patched_axisgrid()
def test_load_seabornobj(tmp_path: Path):
    iris = sns.load_dataset("iris")
    tips = sns.load_dataset("tips")

    # An lmplot
    g0 = sns.lmplot(
        x="total_bill", y="tip", hue="smoker", data=tips, palette=dict(Yes="g", No="m")
    )
    g0 = pw.load_seaborngrid(g0, label="g0")

    # A Pairplot
    g1 = sns.pairplot(iris, hue="species")
    g1 = pw.load_seaborngrid(g1, label="g1", figsize=(6, 6))

    # A relplot
    g2 = sns.relplot(
        data=tips,
        x="total_bill",
        y="tip",
        col="time",
        hue="time",
        size="size",
        style="sex",
        palette=["b", "r"],
        sizes=(10, 100),
    )
    g2.set_titles("")
    g2 = pw.load_seaborngrid(g2, label="g2")

    # A JointGrid
    g3 = sns.jointplot(
        data=iris, x="sepal_width", y="petal_length", kind="kde", space=0
    )
    g3 = pw.load_seaborngrid(g3, label="g3")

    composite = (((g0/g3)["g0"]|g1)["g1"]/g2).savefig()

    result_file = tmp_path / "composite.png"
    composite.savefig(result_file)
    assert result_file.exists()


def test_patched_axisgrid():
    with pw.patched_axisgrid():
        assert hasattr(sns.axisgrid.Grid, "_figure")
        assert sns.axisgrid.FacetGrid.add_legend is pw.modified_grid.add_legend

    assert not hasattr(sns.axisgrid.Grid, "_figure")
    assert sns.axisgrid.FacetGrid.add_legend is not pw.modified_grid.add_legend


def test_patched_plotnine():
    with pw.patched_plotnine():
        if pw.patchworklib._needs_plotnine_ggplot_draw_patch:
            assert p9.ggplot.draw is pw.modified_plotnine.draw

    assert p9.ggplot.draw is not pw.modified_plotnine.draw
