# CSV + PNG metric logging for training. Master-rank-only usage enforced by caller.
#
# Why CSV is the source of truth: training writes one row per --log_interval; PNGs
# are derived from the CSV by re-reading on each refresh. That keeps the trainer
# loop cheap (a single fwrite) and lets a re-plot script regenerate graphs from
# any partial run.

from __future__ import annotations

import os
from typing import Optional

CSV_HEADER = "step,loss,loss_avg,lr,grad_norm,batch_time\n"


class MetricsCsvWriter:
    """Append-only CSV of training scalars; line-buffered so `tail -f` works."""

    def __init__(self, csv_path: str) -> None:
        self.csv_path = csv_path
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        new_file = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
        # buffering=1 == line-buffered in text mode.
        self._fh = open(csv_path, "a", buffering=1, encoding="utf-8")
        if new_file:
            self._fh.write(CSV_HEADER)

    def append(
        self,
        step: int,
        loss: float,
        loss_avg: float,
        lr: float,
        grad_norm: float,
        batch_time: float,
    ) -> None:
        self._fh.write(
            f"{int(step)},{float(loss):.6f},{float(loss_avg):.6f},"
            f"{float(lr):.10f},{float(grad_norm):.6f},{float(batch_time):.6f}\n"
        )

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


class MetricsPngPlotter:
    """Reads the CSV and renders 4 single-metric PNGs + a 2x2 dashboard.

    Stateless w.r.t. trainer running averages: every refresh re-reads the file.
    Failure mode: any matplotlib error logs once and is suppressed afterwards.
    """

    def __init__(self, csv_path: str, out_dir: str) -> None:
        self.csv_path = csv_path
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self._disabled = False
        self._warned = False
        try:
            import matplotlib  # noqa: F401
            matplotlib.use("Agg", force=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[train_logging] matplotlib unavailable; PNG plots disabled: {exc}")
            self._disabled = True

    def _read_csv(self):
        import numpy as np

        if not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) <= len(CSV_HEADER):
            return None
        try:
            arr = np.genfromtxt(
                self.csv_path,
                delimiter=",",
                names=True,
                dtype=None,
                encoding="utf-8",
                invalid_raise=False,
            )
        except Exception as exc:  # noqa: BLE001
            if not self._warned:
                print(f"[train_logging] CSV parse error (suppressing further): {exc}")
                self._warned = True
            return None
        # genfromtxt returns 0-d for single-row files.
        if arr.ndim == 0:
            arr = arr.reshape(1)
        if arr.size == 0:
            return None
        return arr

    @staticmethod
    def _atomic_save(fig, out_path: str) -> None:
        tmp = out_path + ".tmp.png"
        fig.savefig(tmp, dpi=110, bbox_inches="tight", format="png")
        os.replace(tmp, out_path)

    def refresh(self) -> None:
        if self._disabled:
            return
        try:
            import matplotlib.pyplot as plt
        except Exception as exc:  # noqa: BLE001
            if not self._warned:
                print(f"[train_logging] pyplot import failed; disabling: {exc}")
                self._warned = True
            self._disabled = True
            return

        arr = self._read_csv()
        if arr is None:
            return

        try:
            steps = arr["step"].astype(float)
            loss = arr["loss"].astype(float)
            loss_avg = arr["loss_avg"].astype(float)
            lr = arr["lr"].astype(float)
            gn = arr["grad_norm"].astype(float)
            bt = arr["batch_time"].astype(float)

            # loss.png
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(steps, loss, label="loss", linewidth=0.8, alpha=0.6)
            ax.plot(steps, loss_avg, label="loss_avg", linewidth=1.4)
            ax.set_xlabel("step"); ax.set_ylabel("loss"); ax.legend(); ax.grid(True, alpha=0.3)
            ax.set_title("Training loss")
            self._atomic_save(fig, os.path.join(self.out_dir, "loss.png"))
            plt.close(fig)

            # lr.png
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(steps, lr, color="tab:orange")
            ax.set_xlabel("step"); ax.set_ylabel("lr"); ax.grid(True, alpha=0.3)
            ax.set_title("Learning rate")
            self._atomic_save(fig, os.path.join(self.out_dir, "lr.png"))
            plt.close(fig)

            # grad_norm.png — skip rows where grad_norm == 0 (non-sync steps).
            mask = gn > 0
            fig, ax = plt.subplots(figsize=(7, 4))
            if mask.any():
                ax.plot(steps[mask], gn[mask], color="tab:red", linewidth=0.9)
            ax.set_xlabel("step"); ax.set_ylabel("grad_norm"); ax.grid(True, alpha=0.3)
            ax.set_title("Gradient norm")
            self._atomic_save(fig, os.path.join(self.out_dir, "grad_norm.png"))
            plt.close(fig)

            # batch_time.png
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(steps, bt, color="tab:green", linewidth=0.9)
            ax.set_xlabel("step"); ax.set_ylabel("batch_time (s)"); ax.grid(True, alpha=0.3)
            ax.set_title("Per-step wall time")
            self._atomic_save(fig, os.path.join(self.out_dir, "batch_time.png"))
            plt.close(fig)

            # dashboard.png — 2x2 combined grid.
            fig, axes = plt.subplots(2, 2, figsize=(13, 8))
            ax = axes[0, 0]
            ax.plot(steps, loss, alpha=0.6, linewidth=0.8, label="loss")
            ax.plot(steps, loss_avg, linewidth=1.4, label="loss_avg")
            ax.set_title("loss"); ax.grid(True, alpha=0.3); ax.legend()
            ax = axes[0, 1]
            ax.plot(steps, lr, color="tab:orange")
            ax.set_title("lr"); ax.grid(True, alpha=0.3)
            ax = axes[1, 0]
            if mask.any():
                ax.plot(steps[mask], gn[mask], color="tab:red", linewidth=0.9)
            ax.set_title("grad_norm"); ax.grid(True, alpha=0.3)
            ax = axes[1, 1]
            ax.plot(steps, bt, color="tab:green", linewidth=0.9)
            ax.set_title("batch_time (s)"); ax.grid(True, alpha=0.3)
            for r in range(2):
                for c in range(2):
                    axes[r, c].set_xlabel("step")
            fig.suptitle("Training dashboard")
            fig.tight_layout()
            self._atomic_save(fig, os.path.join(self.out_dir, "dashboard.png"))
            plt.close(fig)
        except Exception as exc:  # noqa: BLE001
            if not self._warned:
                print(f"[train_logging] plotting failed (suppressing further): {exc}")
                self._warned = True


class TrackioRun:
    """Thin wrapper around the trackio SDK with a no-op fallback.

    `self._mod` is the single source of truth for "logger live": None means
    every public method is a no-op. The trainer never has to branch — it can
    blindly call .log / .log_histogram / .finish.

    All SDK calls below are wrapped in `except Exception` because telemetry
    must not crash training; on any failure we drop `_mod` and downgrade to a
    no-op for the rest of the run.
    """

    def __init__(
        self,
        project: str,
        name: str,
        config: Optional[dict] = None,
        space_id: Optional[str] = None,
        disable: bool = False,
    ) -> None:
        self._mod = None
        if disable:
            return
        try:
            import trackio as _tr  # type: ignore
        except Exception as exc:  # noqa: BLE001
            print(f"[train_logging.trackio] not available; logging disabled: {exc}")
            return
        if _tr is None:  # monkeypatched-to-None in tests
            return
        try:
            init_kwargs = {"project": project, "name": name, "config": config or {}}
            if space_id:
                init_kwargs["space_id"] = space_id
            _tr.init(**init_kwargs)
        except Exception as exc:  # noqa: BLE001
            print(f"[train_logging.trackio] init failed; logging disabled: {exc}")
            return
        self._mod = _tr

    @property
    def enabled(self) -> bool:
        return self._mod is not None

    def log(self, metrics: dict, step: Optional[int] = None) -> None:
        mod = self._mod
        if mod is None:
            return
        try:
            mod.log(metrics, step=step)
        except Exception as exc:  # noqa: BLE001
            print(f"[train_logging.trackio] log failed (disabling): {exc}")
            self._mod = None

    def log_histogram(self, key: str, values, step: Optional[int] = None) -> None:
        """Best-effort histogram log. trackio's histogram object varies by
        version; if unavailable we log summary statistics under
        <key>_mean/std/min/max.
        """
        mod = self._mod
        if mod is None:
            return
        try:
            import numpy as np
            arr = np.asarray(values, dtype=float).ravel()
            if arr.size == 0:
                return
            payload: dict = {}
            Hist = getattr(mod, "Histogram", None)
            if Hist is not None:
                try:
                    payload[key] = Hist(arr.tolist())
                except Exception as exc:  # noqa: BLE001
                    print(f"[train_logging.trackio] Histogram() failed; falling back to stats: {exc}")
                    Hist = None
            if Hist is None:
                payload[f"{key}_mean"] = float(arr.mean())
                payload[f"{key}_std"] = float(arr.std())
                payload[f"{key}_min"] = float(arr.min())
                payload[f"{key}_max"] = float(arr.max())
            mod.log(payload, step=step)
        except Exception as exc:  # noqa: BLE001
            print(f"[train_logging.trackio] histogram log failed (disabling): {exc}")
            self._mod = None

    def finish(self) -> None:
        mod = self._mod
        if mod is None:
            return
        self._mod = None
        try:
            mod.finish()
        except Exception as exc:  # noqa: BLE001
            print(f"[train_logging.trackio] finish failed: {exc}")


def make_logger(
    output_dir: str,
    csv_path: Optional[str],
    plot_interval: int,
) -> tuple[MetricsCsvWriter, Optional[MetricsPngPlotter]]:
    """Convenience constructor used by both train_cvlm.py and train_sft.py."""
    path = csv_path.strip() if csv_path else ""
    if not path:
        path = os.path.join(output_dir, "metrics.csv")
    csv_writer = MetricsCsvWriter(path)
    plotter = MetricsPngPlotter(path, out_dir=output_dir) if plot_interval > 0 else None
    return csv_writer, plotter
