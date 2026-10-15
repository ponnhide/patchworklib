# patchworklib API Guide

This guide describes the implementation on GitHub `main`.
A `Brick` is one Matplotlib Axes. A `Bricks` object groups several Axes into a
layout. Draw each panel first, combine the panel objects, and export the result.

- [Quick Start](#quick-start)
- [Layout and Coordinates](#layout-and-coordinates)
- [Layout Functions](#layout-functions)
- [Brick and cBrick](#brick-and-cbrick)
- [Bricks](#bricks)
- [Labels, Legends, and Colorbars](#labels-legends-and-colorbars)
- [Saving and Reusing Panels](#saving-and-reusing-panels)
- [Seaborn Integration](#seaborn-integration)
- [Settings and Cleanup](#settings-and-cleanup)
- [Troubleshooting](#troubleshooting)

## Quick Start

```python
import matplotlib.pyplot as plt
import patchworklib as pw

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})

left = pw.Brick(label="left", figsize=(3, 2))
left.plot([0, 1, 2], [0.1, 0.4, 0.2], color="#0072B2")
left.set_xlabel("Position")
left.set_ylabel("Frequency")

right = pw.Brick(label="right", figsize=(3, 2))
right.bar(["A", "B", "C"], [4, 7, 2], color="#D55E00")
right.set_ylabel("Count")

figure = left | right
left.set_index("A", fontsize=11, fontweight="bold")
right.set_index("B", fontsize=11, fontweight="bold")
figure.savefig("figure.png", dpi=300)
figure.savefig("figure.pdf")
```

Plotnine is not required. Use ordinary Matplotlib Axes methods on a `Brick`.
Seaborn axes-level functions also accept a `Brick` through their `ax` argument.

## Layout and Coordinates

### Operators

Both operands may be either `Brick` or `Bricks` objects.

| Expression | Order | Size adjustment | Equivalent function |
| --- | --- | --- | --- |
| `a \| b` | a left, b right | Match heights | `hstack(a, b)` |
| `a + b` | a left, b right | Keep input sizes | `hstack(a, b, adjust_height=False)` |
| `a / b` | a above b | Match widths | `vstack(b, a)` |
| `a - b` | a above b | Keep input sizes | `vstack(b, a, adjust_width=False)` |

Use parentheses in mixed expressions. For example, `(a | b) / c` puts a and b
in the top row and c below them.

The argument order of `vstack()` differs from the visual order of `/`:
`vstack(a, b)` places **b above a**. To place a above b, use `a / b`,
`vstack(a, b, direction="b")`, or `stack([a, b], operator="/")`.

### Size, Spacing, and Fonts

- `Brick(figsize=(width, height))` specifies the data Axes size in inches.
  Labels and legends occupy additional space outside these dimensions.
- `margin` is a gap in the layout coordinate system. Normal stacking accounts
  for tick labels, axis labels, and other visible decorations.
- Matplotlib text settings such as `fontsize`, title `pad`, and axis
  `labelpad` use points.
- Add panel labels, titles, and legends before composition when possible, so
  their extents can be included in the layout calculation.
- `change_plotsize()` changes the layout. Its requested dimensions do not
  guarantee an identical physical page size when exporting with a tight box.

`margin=None` uses `pw.param["margin"]`; it does not mean zero spacing.
`margin=0` adds no gap between the decorated panel boundaries.
`margin="none"` joins the Axes boundaries without reserving space for their
outside decorations. The latter may overlap tick labels or axis labels.

### Labels and Shared State

A `label` is a unique panel identifier, not a displayed title. If omitted, it
is generated automatically. Creating a second `Brick` with an existing label
raises an error.

Composition changes the positions of the original Axes. Panels are shared
objects, not immutable snapshots. Build fresh panels when editing several
independent layouts. See [Saving and Reusing Panels](#saving-and-reusing-panels)
for independent Figure exports.

## Layout Functions

### `hstack()`

```python
pw.hstack(
    brick1, brick2, target=None, margin=None, direction="r",
    adjust_height=True, adjust_width=True, va="top",
)
```

Place `brick2` to the right or left of `brick1`. Return a new `Bricks` object
containing both panels.

| Parameter | Meaning |
| --- | --- |
| `brick1`, `brick2` | A `Brick` or `Bricks` object. `brick1` supplies the reference position. |
| `target` | The label of a panel inside `brick1` to use as the placement target. By default, use the whole object. |
| `margin` | Gap between the decorated boundaries. `None` uses the global setting. |
| `direction` | `"r"` for right; `"l"` for left. |
| `adjust_height` | Match the height of `brick2` to the reference height. |
| `adjust_width` | Scale the width along with a height adjustment. If `adjust_height=False`, the original width is also retained. |
| `va` | Vertical alignment when heights are not matched: `"top"` or `"bottom"`. The aliases `"t"` and `"b"` are also accepted. |

```python
row = pw.hstack(left, right, margin=0.4, adjust_height=False, va="top")
```

### `vstack()`

```python
pw.vstack(
    brick1, brick2, target=None, margin=None, direction="t",
    adjust_height=True, adjust_width=True, ha="l",
)
```

Place `brick2` above or below `brick1`. Return a new `Bricks` object containing
both panels.

| Parameter | Meaning |
| --- | --- |
| `brick1`, `brick2`, `target`, `margin` | The same roles as in `hstack()`. |
| `direction` | `"t"` places `brick2` above; `"b"` places it below. |
| `adjust_width` | Match the width of `brick2` to the reference width. |
| `adjust_height` | Scale the height along with a width adjustment. If `adjust_width=False`, the original height is also retained. |
| `ha` | Horizontal alignment when widths are not matched: `"l"` or `"left"`, and `"r"` or `"right"`. |

```python
# Preserve input sizes and place bottom below top.
column = pw.vstack(top, bottom, direction="b", adjust_width=False, margin=0.4)
```

### `stack()`

```python
pw.stack(bricks, margin=None, operator="|", equal_spacing=False)
```

Arrange a list of `Brick` and/or `Bricks` objects in input order.
Horizontal operators `"|"` and `"+"` run left to right; vertical operators
`"/"` and `"-"` run top to bottom. `"|"` matches heights, `"/"` matches widths,
and `"+"` and `"-"` retain the corresponding input sizes.

With `equal_spacing=True`, equalize gaps between the inner Axes boundaries.
Otherwise, the decorated panel boundaries determine placement. The return
value is normally `Bricks`; a one-element list returns that input object.
Supply a nonempty list. A single panel does not need `equal_spacing=True`.

```python
row = pw.stack([a, b, c], operator="|", margin=0.3)
```

### `inset()`

```python
pw.inset(
    brick1, brick2, loc="upper right", wratio=0.4, hratio=0.4,
    vmargin=0.1, hmargin=0.1, alpha=0.0,
)
```

Place `brick2` inside `brick1` and return a `Bricks` object containing both.

`loc` accepts `"upper right"`, `"upper left"`, `"lower right"`, or
`"lower left"`. The keyword is `loc`, not `position`. `wratio` and `hratio`
set the inset width and height relative to the host. `hmargin` and `vmargin`
are distances in layout coordinates; `alpha` controls the inset background
opacity.

### `expand()` and `spacer()`

```python
pw.expand(bricks, w, h)
pw.spacer(brick=None, value=1.0)
```

`expand()` scales the width by `w` and height by `h`, modifies the supplied
layout, and returns it. These arguments are scale factors, not target sizes.

`spacer()` creates blank space relative to a reference panel. Combine it with
`|` or `/`. For example, `pw.spacer(left, value=0.5) | right` puts blank space
based on left's dimensions beside right.

## Brick and cBrick

```python
pw.Brick(label=None, figsize=None, ax=None)
pw.cBrick(label=None, figsize=None, ax=None)
```

`Brick` subclasses `matplotlib.axes.Axes`; `cBrick` subclasses the polar Axes.
A newly created panel defaults to `figsize=(1, 1)`. An existing Matplotlib
Axes can be supplied as `ax`. This conversion shares state with the original
Axes and should not be treated as an independent copy.

```python
polar = pw.cBrick(label="polar", figsize=(2, 2))
polar.plot([0, 1, 2], [1, 2, 1])
```

Use ordinary Matplotlib methods such as `plot()`, `bar()`, `imshow()`,
`set_xlim()`, `set_xticks()`, `set_title()`, `set_xlabel()`, `set_ylabel()`,
and `text()`. For Axes-relative text coordinates, pass
`transform=brick.transAxes` to `text()`.

| Attribute or method | Meaning |
| --- | --- |
| `label` / `get_label()` | Unique identifier, distinct from the displayed title. |
| `bricks_dict` | A dictionary containing this panel under its label. |
| `case` | An invisible Axes surrounding the panel and its axis decorations. |
| `outline` | A `Bricks` object representing the outer extent, including decorations placed on case. |
| `reset_label(name)` | Change the identifier. |
| `change_plotsize(new_size)` | Set the data Axes size to `(width, height)` inches. Return `None`. |
| `change_aspectratio(new_size)` | Deprecated name; use `change_plotsize()`. |
| `set_index(index, x=None, y=None, **args)` | Add a panel letter on case, at the upper left by default. Return a `Text`. |
| `add_text(x, y, text, **args)` | Add text in case-relative coordinates. Return a `Text`. |
| `move_legend(new_loc, **kws)` | Recreate an existing legend at a new location. Create the legend first. |

`Brick` does **not** accept `bricks_dict=` in its constructor; that parameter
belongs to `Bricks`. The case-text helper is `Brick.add_text()`, whereas the
corresponding `Bricks` helper is `set_text()`.

## Bricks

```python
pw.Bricks(bricks_dict=None, label=None)
```

Group existing Axes. Most code obtains this object from a layout operator.
For direct construction, supply a nonempty dictionary mapping labels to
`Brick` objects. Although the signature contains a `None` default, calling
`Bricks()` without a dictionary does not create a usable empty layout.

```python
group = pw.Bricks({"left": left, "right": right}, label="group")
```

The constructor groups the existing positions; it does not arrange the Axes.
Use layout operators or stacking functions to position panels.

Access the data Axes through `group.bricks_dict`. Label indexing also selects
a placement target: `group["left"] | extra` places extra beside left and
returns a layout containing the whole group. For simple styling changes,
use the original `left` variable or `group.bricks_dict["left"]` to avoid
selecting a placement target.

| Attribute or method | Meaning |
| --- | --- |
| `label` / `get_label()` | Identifier for this group. |
| `bricks_dict` | Dictionary of data Axes. Common-label case axes are not included. |
| `case` | Invisible Axes for common labels, legends, and annotations. |
| `outline` | A `Bricks` object representing the extent including common decorations. |
| `reset_label(name)` | Change the group identifier. |
| `change_plotsize(new_size)` | Resize the composition to the requested width and height. Return `None`. Font sizes are not scaled automatically. |
| `set_originalpositions()` | Record the current positions for subsequent exports of this group. Use after directly changing Axes positions; normal composition operations record them internally. |
| `align_xlabels(keys=None)` | Align x-axis labels of selected panels. |
| `align_ylabels(keys=None, ha="left")` | Align y-axis labels of selected panels. |

`keys` accepts a list of labels or panel objects; omit it to use panels in the
group. A `Bricks` object is not an Axes, so it has no `plot()` or `text()` method.

## Labels, Legends, and Colorbars

### Common Labels and Annotations

```python
group.set_supxlabel(xlabel, labelpad=None, *, loc=None, **args)
group.set_supylabel(ylabel, labelpad=None, *, loc=None, **args)
group.set_suptitle(title, loc=None, pad=None, **args)
group.set_index(index, x=None, y=None, **args)
group.set_text(x, y, text, **args)
```

These methods return a `matplotlib.text.Text` object. Additional keywords
include Matplotlib text properties such as `fontsize`, `fontweight`, and
`color`. Accepted `loc` values follow the corresponding Matplotlib label or
title method. `set_text()` uses case-relative coordinates.

When omitted, `labelpad` is `plt.rcParams["axes.labelpad"] + 4`, and title
`pad` is the same setting plus 8. Finalize text and fonts before arranging
panels when possible.

The existing methods for moving common axis labels to another side have the
following spelling:

```python
group.set_supxlabel_positon("top")
group.set_supylabel_positon("right")
```

The spelling is `positon`, not `position`. It is retained for API compatibility.

### Common Spines

```python
group.set_supspine(which="left", visible=True, position=None, bounds=None)
```

Configure a case spine and return its `Spine` object. `which` is `"left"`,
`"right"`, `"top"`, or `"bottom"`. `position` and `bounds` follow Matplotlib's
Spine API.

### Legends

Both `Brick` and `Bricks` provide `move_legend(new_loc, **kws)`. This method
moves an existing legend; it does not collect handles or create a new legend
from the group. For a shared legend, first call
`group.case.legend(handles, labels, ...)`.

```python
left.plot([0, 1], [0, 1], label="Sample")
left.legend()
left.move_legend("upper left", bbox_to_anchor=(1.02, 1.0))
```

### Colorbars

```python
panel.add_colorbar(
    cmap=None, x=None, y=None, vmin=0, vmax=1,
    hratio=None, wratio=None, coordinate="relative", label=None, **args,
)
```

Available on both `Brick` and `Bricks`. It returns a **new `Bricks` object
including the colorbar**. Use that return value for composition and export.

```python
heatmap = pw.Brick(label="heatmap", figsize=(2, 2))
heatmap.imshow([[0.1, 0.4], [0.7, 1.0]], cmap="viridis", vmin=0, vmax=1)
with_colorbar = heatmap.add_colorbar(cmap="viridis", vmin=0, vmax=1)
```

| Parameter | Meaning |
| --- | --- |
| `cmap` | Matplotlib colormap, viridis by default. Match the plot's colormap. |
| `vmin`, `vmax` | Minimum and maximum scale values. |
| `x`, `y` | Offsets from the lower-left reference position. Omit for automatic placement appropriate to the orientation. |
| `coordinate` | `"relative"` interprets offsets as fractions of the reference dimensions; `"absolute"` uses layout distances. |
| `hratio`, `wratio` | Height and width relative to the reference dimensions. Defaults depend on orientation. |
| `label` | Unique identifier for the colorbar Axes, not its displayed scale label. |
| `**args` | Arguments forwarded to Matplotlib ColorbarBase, such as `orientation="horizontal"`. |

### Inspecting Panel Bounds

`get_inner_corner()`, `get_middle_corner()`, and `get_outer_corner()` are
available on both `Brick` and `Bricks`. Each returns
`(left, right, bottom, top)`, **not** `(x, y, width, height)`.

Inner bounds describe the data Axes; middle bounds include axis decorations;
outer bounds also include decorations placed on case.

## Saving and Reusing Panels

```python
panel.savefig(fname=None, transparent=False, quick=True, **kwargs)
```

Available on both `Brick` and `Bricks`. `fname` accepts a filename, a Path,
or a writable buffer. Keywords such as `format`, `dpi`, and `bbox_inches`
are forwarded to Matplotlib. The defaults are `bbox_inches="tight"` and
`dpi=pw.param["dpi"]`. The internal `_ggplot` argument is retained for
compatibility and is not needed in normal code.

| Call | Result and shared state |
| --- | --- |
| `savefig("figure.png")` | Write only this panel/composition and return the shared Matplotlib Figure. Temporary export positioning is restored afterward, including on failure. |
| `savefig(quick=False)` | Return an independent Matplotlib Figure containing only this panel/composition. Restore the original Axes positions after making the copy. |
| `savefig()` | Return the shared Figure for a quick preview. Later layout or export operations can change it. |

Other panels remain usable for plotting and annotation after a quick export.
File exports preserve the live positions in parent layouts. Explicitly
resizing or recomposing an object still changes its shared Axes.

### Exporting Several Formats with One Layout

Capture an independent Figure once, then save that same Figure in each format.

```python
import matplotlib.pyplot as plt

plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42

rendered = figure.savefig(quick=False)
for extension in ("pdf", "svg", "png"):
    rendered.savefig(f"figure.{extension}", bbox_inches="tight", dpi=300)
```

`svg.fonttype="none"` retains SVG text instead of converting glyphs to paths.
An `imshow()` panel still contains image content; this setting does not make
raster artists into vectors. Font metrics may differ slightly among output
backends.

`change_plotsize((7.2, 4.0))` does not guarantee a tight-export page of exactly
7.2 by 4.0 inches, because the page also includes labels and other decorations.

### Saving Panels Before Composition

```python
left.savefig("left.png")
figure = left | right
figure.savefig("combined.png")
```

Saving a panel that is already part of another layout preserves that parent
layout's Axes positions. To retain an independent exported Figure, use
`quick=False`. After calling `pw.clear()`, create new panels instead of reusing
the old objects.

## Seaborn Integration

### Axes-Level Functions

Pass `ax=brick` to functions such as `sns.scatterplot()`, `sns.lineplot()`,
and `sns.barplot()`. No conversion is needed.

```python
import pandas as pd
import seaborn as sns

data = pd.DataFrame({"x": [0, 1, 2, 0, 1, 2], "y": [1, 3, 2, 2, 4, 3]})
axis = pw.Brick(label="seaborn_axis", figsize=(3, 2))
sns.lineplot(data=data, x="x", y="y", ax=axis)
```

### Figure-Level Functions

```python
pw.overwrite_axisgrid()
pw.load_seaborngrid(g, label=None, labels=None, figsize=None)
```

`overwrite_axisgrid()` changes Seaborn Grid classes in the current process to
use patchworklib's shared Figure. Call it **before** functions such as
`relplot()`, `catplot()`, `pairplot()`, or `jointplot()`.

Pass the returned grid to `load_seaborngrid()`. The result is a `Brick` or
`Bricks`, depending on the number of Axes.

```python
pw.overwrite_axisgrid()
grid = sns.relplot(data=data, x="x", y="y", kind="scatter")
loaded = pw.load_seaborngrid(grid, label="loaded_grid", figsize=(3, 2))
loaded.savefig("seaborn_grid.png")
```

`label` identifies the returned panel/group. `labels` supplies identifiers for
multiple internal Axes. `figsize=None` uses the source grid's size information.
The integration depends on Seaborn internals, so check the Seaborn version
used in your environment. `pw.clear()` does not undo `overwrite_axisgrid()`.

### Seaborn Objects and Legacy Plotnine Functions

```python
pw.load_seabornobj(g, label=None, labels=None, figsize=(3, 3))
```

This experimental adapter reads a rendered Plotter from the Seaborn Objects
interface. Use `load_seaborngrid()` for ordinary FacetGrid and related grids.
The adapter depends on private Seaborn attributes; prefer axes-level functions
for new workflows when possible.

`load_ggplot(ggplot=None, figsize=None)` and `overwrite_plotnine()` remain as
legacy implementations. Plotnine support ended in 0.6.6; compatibility with
current plotnine versions is not guaranteed. Plotnine is not required for
ordinary Matplotlib and Seaborn composition.

## Settings and Cleanup

```python
pw.param["margin"] = 0.5
pw.param["dpi"] = 200
pw.clear()
```

`param` is shared configuration. To change only one composition's spacing,
pass `margin=` to its stacking function instead of modifying the global value.

`clear()` removes the existing panels and label registrations. Use it when
starting a new independent analysis or test. It does not reset Matplotlib
`rcParams`, `pw.param`, or the Seaborn Grid changes.

`basefigure` and `axes_dict` expose shared Figure/registry state. Ordinary
composition code does not need to access them directly.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| Vertical order is reversed | `vstack(a, b)` puts b above a. For input order, use `stack([a, b], operator="/")`. |
| `margin=None` leaves a gap | It selects the global margin. Use `margin=0` for no additional gap. |
| A `Bricks` object has no `text()` or `set_xlabel()` | Use the original `Brick` for Axes methods, or `set_text()` / `set_supxlabel()` for a group. |
| Duplicate-label error | Use a unique `label`, or call `clear()` when the previous panels are no longer needed. |
| A colorbar is missing from the export | Save the object returned by `add_colorbar()`. |
| Exported page is larger than the requested size | Data Axes dimensions and the decorated tight-export page dimensions differ. |
| A previously returned Figure changes | `quick=True` returns a shared Figure. Use `quick=False` to retain an independent copy. |
| Annotation after saving raises `NoneType ... dpi` | This was an export-state bug in older code. Use a GitHub revision containing the export-state fix. |
| An old plotnine example fails | Plotnine support ended in 0.6.6. Use the supported Matplotlib/Seaborn paths. |
