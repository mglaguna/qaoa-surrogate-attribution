"""Raise every text element smaller than MIN_PT points to MIN_PT just before a figure is
saved, without changing any plotted data. Import this module before running a figure
script, or use `python enlarge_fonts.py SCRIPT [ARGS...]`.
"""
import runpy, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.figure as _F
import matplotlib.text as _T

MIN_PT = 8.0
_orig = _F.Figure.savefig


def _savefig(self, *args, **kwargs):
    for t in self.findobj(_T.Text):
        try:
            if t.get_text() and t.get_fontsize() < MIN_PT:
                t.set_fontsize(MIN_PT)
        except Exception:
            pass
    return _orig(self, *args, **kwargs)


_F.Figure.savefig = _savefig


def run(script, argv):
    sys.argv = [script] + list(argv)
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2:])
