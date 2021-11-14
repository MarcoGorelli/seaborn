from __future__ import annotations
import numpy as np
from matplotlib.colors import to_rgba
from seaborn._compat import MarkerStyle
from seaborn._marks.base import Mark, Feature


class Point(Mark):  # TODO types

    supports = ["color"]

    def __init__(
        self,
        color=Feature("C0"),
        edgecolor=Feature("w"),
        marker=Feature(rc="scatter.marker"),
        pointsize=Feature(4),
        linewidth=Feature(1),  # TODO how to scale with point size?
        edgewidth=Feature(.5),  # TODO how to scale with point size?
        fill=Feature(True),
        jitter=None,
        **kwargs,  # TODO needed?
    ):

        super().__init__(**kwargs)

        self.features = {k: v for k, v in locals().items() if isinstance(v, Feature)}

        self.jitter = jitter  # TODO decide on form of jitter and add type hinting

    def _adjust(self, df, mappings, orient):

        if self.jitter is None:
            return df

        x, y = self.jitter  # TODO maybe not format, and do better error handling

        # TODO maybe accept a Jitter class so we can control things like distribution?
        # If we do that, should we allow convenient flexibility (i.e. (x, y) tuple)
        # in the object interface, or be simpler but more verbose?

        # TODO note that some marks will have multiple adjustments
        # (e.g. strip plot has both dodging and jittering)

        # TODO native scale of jitter? maybe just for a Strip subclass?

        rng = np.random.default_rng()  # TODO seed?

        n = len(df)
        x_jitter = 0 if not x else rng.uniform(-x, +x, n)
        y_jitter = 0 if not y else rng.uniform(-y, +y, n)

        # TODO: this fails if x or y are paired. Apply to all columns that start with y?
        return df.assign(x=df["x"] + x_jitter, y=df["y"] + y_jitter)

    def _plot_split(self, keys, data, ax, orient, kws):

        # TODO can we simplify this by modifying data with mappings before sending in?
        # Likewise, will we need to know `keys` here? Elsewhere we do `if key in keys`,
        # but I think we can (or can make it so we can) just do `if key in data`.

        # Then the signature could be _plot_split(ax, data, kws):  ... much simpler!

        # TODO Not backcompat with allowed (but nonfunctional) univariate plots

        kws = kws.copy()

        color = self._resolve("color", mappings, data, to_rgba)
        edgecolor = self._resolve("edgecolor", mappings, data, to_rgba)
        marker = self._resolve("marker", mappings, data, MarkerStyle)
        fill = self._resolve("fill", mappings, data, bool)

        # TODO matplotlib has "edgecolor='face'" and it would be good to keep that
        # But it would be BETTER to have succient way of specifiying, e.g.
        # edgecolor = set_hls_values(facecolor, l=.8)

        # TODO lots of questions about the best way to implement fill
        # e.g. we need to remap color to edgecolor where edge is false
        fill &= np.array([m.is_filled() for m in marker])
        edgecolor[~fill] = color[~fill]
        color[~fill] = 0

        points = ax.scatter(x=data["x"], y=data["y"], **kws)
        points.set_facecolors(color)
        points.set_edgecolors(edgecolor)

        paths = [m.get_path().transformed(m.get_transform()) for m in marker]
        points.set_paths(paths)

        # TODO this doesn't work. Apparently scatter is reading
        # the marker.is_filled attribute and directing colors towards
        # the edge/face and then setting the face to uncolored as needed.
        # We are getting to the point where just creating the PathCollection
        # ourselves is probably easier, but not breaking existing scatterplot
        # calls that leverage ax.scatter features like cmap might be tricky.
        # Another option could be to have some internal-only Marks that support
        # the existing functional interface where doing so through the new
        # interface would be overly cumbersome.
        # Either way, it would be best to have a common function like
        # apply_fill(facecolor, edgecolor, filled)
        # We may want to think about how to work with MarkerStyle objects
        # in the absence of a `fill` semantic so that we can relax the
        # constraint on mixing filled and unfilled markers...


class Line(Mark):

    # TODO how to handle distinction between stat groupers and plot groupers?
    # i.e. Line needs to aggregate by x, but not plot by it
    # also how will this get parametrized to support orient=?
    # TODO will this sort by the orient dimension like lineplot currently does?
    grouping_vars = ["color", "marker", "linestyle", "linewidth"]
    supports = ["color", "marker", "linestyle", "linewidth"]

    def _plot_split(self, keys, data, ax, orient, kws):

        if "color" in keys:
            kws["color"] = self.mappings["color"](keys["color"])
        if "linestyle" in keys:
            kws["linestyle"] = self.mappings["linestyle"](keys["linestyle"])
        if "linewidth" in keys:
            kws["linewidth"] = self.mappings["linewidth"](keys["linewidth"])

        ax.plot(data["x"], data["y"], **kws)


class Area(Mark):

    grouping_vars = ["color"]
    supports = ["color"]

    def _plot_split(self, keys, data, ax, orient, kws):

        if "color" in keys:
            # TODO as we need the kwarg to be facecolor, that should be the mappable?
            kws["facecolor"] = self.mappings["color"](keys["color"])

        # TODO how will orient work here?
        # Currently this requires you to specify both orient and use y, xmin, xmin
        # to get a fill along the x axis. Seems like we should need only one of those?
        # Alternatively, should we just make the PolyCollection manually?
        if orient == "x":
            ax.fill_between(data["x"], data["ymin"], data["ymax"], **kws)
        else:
            ax.fill_betweenx(data["y"], data["xmin"], data["xmax"], **kws)
