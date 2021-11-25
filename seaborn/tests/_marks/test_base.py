
import numpy as np
import pandas as pd
import matplotlib as mpl

from numpy.testing import assert_array_equal

from seaborn._marks.base import Mark, Feature


class TestFeatures:

    def mark(self, **features):

        m = Mark()
        m.features = features
        return m

    def test_value(self):

        val = 3
        m = self.mark(linewidth=val)
        assert m._resolve({}, "linewidth") == val

        df = pd.DataFrame(index=pd.RangeIndex(10))
        assert_array_equal(m._resolve(df, "linewidth"), np.full(len(df), val))

    def test_default(self):

        val = 3
        m = self.mark(linewidth=Feature(val))
        assert m._resolve({}, "linewidth") == val

        df = pd.DataFrame(index=pd.RangeIndex(10))
        assert_array_equal(m._resolve(df, "linewidth"), np.full(len(df), val))

    def test_rcparam(self):

        param = "lines.linewidth"
        val = mpl.rcParams[param]

        m = self.mark(linewidth=Feature(rc=param))
        assert m._resolve({}, "linewidth") == val

        df = pd.DataFrame(index=pd.RangeIndex(10))
        assert_array_equal(m._resolve(df, "linewidth"), np.full(len(df), val))

    def test_depends(self):

        val = 2
        df = pd.DataFrame(index=pd.RangeIndex(10))

        m = self.mark(pointsize=Feature(val), linewidth=Feature(depend="pointsize"))
        assert m._resolve({}, "linewidth") == val
        assert_array_equal(m._resolve(df, "linewidth"), np.full(len(df), val))

        m = self.mark(pointsize=val * 2, linewidth=Feature(depend="pointsize"))
        assert m._resolve({}, "linewidth") == val * 2
        assert_array_equal(m._resolve(df, "linewidth"), np.full(len(df), val * 2))
