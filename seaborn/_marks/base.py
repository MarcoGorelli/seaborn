from __future__ import annotations
from contextlib import contextmanager

import numpy as np
import pandas as pd
import matplotlib as mpl

from seaborn._core.plot import SEMANTICS

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Literal, Any, Type, Dict, Callable
    from collections.abc import Generator
    from pandas import DataFrame
    from matplotlib.axes import Axes
    from seaborn._core.mappings import SemanticMapping
    from seaborn._stats.base import Stat

    MappingDict = Dict[str, SemanticMapping]


class Feature:
    """Class supporting visual features set directly, via mapping, or from rcParams."""
    def __init__(
        self,
        val: Any = None,
        depend: str | None = None,
        rc: str | None = None
    ):
        self.val = val
        self.depend = depend
        self.rc = rc

        # TODO some sort of smart=True default to indicate that default value is
        # dependent the specific plot?

    @property
    def default(self) -> Any:
        if self.val is not None:
            return self.val
        return mpl.rcParams.get(self.rc)

    # TODO make immutable, this gets cached in docstrings
    # TODO nice str/repr for docstring


class Mark:
    """Base class for objects that control the actual plotting."""
    # TODO where to define vars we always group by (col, row, group)
    default_stat: Type[Stat] | None = None
    grouping_vars: list[str] = []
    requires: list[str]  # List of variabes that must be defined
    supports: list[str]  # List of variables that will be used
    features: dict[str, Any]

    def __init__(self, **kwargs: Any):

        self.features = {}
        self._kwargs = kwargs

    @contextmanager
    def use(self, mappings, orient) -> None:

        self.mappings = mappings
        self.orient = orient
        try:
            yield
        finally:
            del self.mappings, self.orient

    def _resolve(
        self,
        data: DataFrame | dict,
        name: str,
    ) -> Any:

        # TODO could we reach into SEMANTICS and use standardization function
        # rather than passing in f? Or even do that in the constructor?
        # (But probably fine to do it here too)

        feature = self.features[name]
        standardize = SEMANTICS[name]._standardize_value
        directly_specified = not isinstance(feature, Feature)

        if directly_specified:
            feature = standardize(feature)
            if isinstance(data, pd.DataFrame):
                feature = np.array([feature] * len(data))
            return feature

        if name in data:
            return np.asarray(self.mappings[name](data[name]))

        if feature.depend is not None:
            # TODO add source_func or similar to transform the source value
            # e.g. set linewidth by pointsize or extract alpha value from color
            # (latter suggests a new concept: "second-order" features/semantics)
            return self._resolve(data, feature.depend)

        default = standardize(feature.default)
        if isinstance(data, pd.DataFrame):
            default = np.array([default] * len(data))
        return default

    def _resolve_color(
        self,
        data: DataFrame | dict,
        prefix: str = "",
    ) -> Any:  # TODO tighter type here? rgba array, right?

        # TODO
        # we want this to respect alpha if set in the {fill/edge}color feature,
        # and otherwise use the default alpha value.
        # (But what if alpha is passed in both color and as explicit alpha?)
        # This is somewhat tricky to do!

        color = self._resolve(data, f"{prefix}color")
        alpha = self._resolve(data, f"{prefix}alpha")

        if isinstance(color, tuple):
            return mpl.colors.to_rgba(color, alpha)
        else:
            return mpl.colors.to_rgba_array(color, alpha)

    def _adjust(
        self,
        df: DataFrame,
    ) -> DataFrame:

        return df

    def _infer_orient(self, scales: dict) -> Literal["x", "y"]:  # TODO type scale

        # TODO The original version of this (in seaborn._oldcore) did more checking.
        # Paring that down here for the prototype to see what restrictions make sense.

        x_type = None if "x" not in scales else scales["x"].scale_type
        y_type = None if "y" not in scales else scales["y"].scale_type

        if x_type is None:
            return "y"

        elif y_type is None:
            return "x"

        elif x_type != "categorical" and y_type == "categorical":
            return "y"

        elif x_type != "numeric" and y_type == "numeric":

            # TODO should we try to orient based on number of unique values?

            return "x"

        elif x_type == "numeric" and y_type != "numeric":
            return "y"

        else:
            return "x"

    def _plot(
        self,
        split_generator: Callable[[], Generator],
    ) -> None:
        """Main interface for creating a plot."""
        for keys, data, ax in split_generator():
            kws = self._kwargs.copy()
            self._plot_split(keys, data, ax, kws)

        self._finish_plot()

    def _plot_split(
        self,
        keys: dict[str, Any],
        data: DataFrame,
        ax: Axes,
        kws: dict,
    ) -> None:
        """Method that plots specific subsets of data. Must be defined by subclass."""
        raise NotImplementedError()

    def _finish_plot(self) -> None:
        """Method that is called after each data subset has been plotted."""
        pass
