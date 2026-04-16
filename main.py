"""
Wave Optics Simulation: Interference & Diffraction
===================================================
Simulates Young's Double Slit, Single Slit Diffraction, and Diffraction Grating
using Fraunhofer (far-field) approximations.

Physics constants & unit conventions used throughout:
  - Wavelength (λ): stored internally in metres (input in nm)
  - Slit separation (d): stored internally in metres (input in mm)
  - Slit width (a): stored internally in metres (input in µm)
  - Screen distance (D): metres
  - Screen position (x): metres, displayed in mm
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

matplotlib.use("TkAgg")

# ─────────────────────────────────────────────────────────────────────────────
# Colour helpers
# ─────────────────────────────────────────────────────────────────────────────

def wavelength_to_rgb(wl_nm: float) -> tuple:
    """
    Convert a visible-light wavelength (380–780 nm) to an approximate sRGB
    triplet.  Outside that range the function returns near-black.
    Based on Dan Bruton's algorithm (www.physics.sfasu.edu/astro/color).
    """
    w = float(wl_nm)
    if 380 <= w < 440:
        r, g, b = -(w - 440) / 60, 0.0, 1.0
    elif 440 <= w < 490:
        r, g, b = 0.0, (w - 440) / 50, 1.0
    elif 490 <= w < 510:
        r, g, b = 0.0, 1.0, -(w - 510) / 20
    elif 510 <= w < 580:
        r, g, b = (w - 510) / 70, 1.0, 0.0
    elif 580 <= w < 645:
        r, g, b = 1.0, -(w - 645) / 65, 0.0
    elif 645 <= w <= 780:
        r, g, b = 1.0, 0.0, 0.0
    else:
        r, g, b = 0.0, 0.0, 0.0

    # Intensity fall-off at edges of visible range
    if 380 <= w < 420:
        factor = 0.3 + 0.7 * (w - 380) / 40
    elif 700 < w <= 780:
        factor = 0.3 + 0.7 * (780 - w) / 80
    else:
        factor = 1.0

    return (r * factor, g * factor, b * factor)


def make_screen_colormap(wl_nm: float) -> mcolors.LinearSegmentedColormap:
    """Build a simple black → wavelength-colour colormap for imshow."""
    colour = wavelength_to_rgb(wl_nm)
    return mcolors.LinearSegmentedColormap.from_list(
        "light", [(0, 0, 0), colour], N=256
    )


# ─────────────────────────────────────────────────────────────────────────────
# Physics kernels  (all angles / positions in SI; intensity normalised to 1)
# ─────────────────────────────────────────────────────────────────────────────

def intensity_double_slit(x, lam, d, a, D):
    """
    Young's double slit with finite slit width (diffraction envelope).

    I(x) = sinc²(β/π) · cos²(α)

    where
      β = π·a·x / (λ·D)   — slit-width phase half-angle
      α = π·d·x / (λ·D)   — slit-separation phase half-angle

    sinc here is the *unnormalised* sinc: sinc(u) = sin(u)/u
    numpy.sinc is the *normalised* sinc: sinc(u) = sin(πu)/(πu)
    so  sinc_unnorm(β) = np.sinc(β/π)
    """
    beta  = np.pi * a * x / (lam * D)   # slit-width phase
    alpha = np.pi * d * x / (lam * D)   # separation phase

    # Diffraction envelope (single-slit factor)
    envelope = np.sinc(beta / np.pi) ** 2   # numpy.sinc(u) = sin(πu)/(πu)

    # Interference fringes
    fringes = np.cos(alpha) ** 2

    return envelope * fringes


def intensity_single_slit(x, lam, a, D):
    """
    Single-slit Fraunhofer diffraction.

    I(x) = sinc²(β/π),   β = π·a·x / (λ·D)
    """
    beta = np.pi * a * x / (lam * D)
    return np.sinc(beta / np.pi) ** 2


def intensity_grating(x, lam, d, N, D):
    """
    Diffraction grating (N slits, grating element d = 1/lines_per_mm).

    I(x) = [sin(β)/β]² · [sin(N·γ)/sin(γ)]²

    where
      β = π·a_eff·x / (λ·D)   — single-slit envelope
                                 (a_eff ≈ d/2 assumed for display purposes)
      γ = π·d·x   / (λ·D)     — grating phase per slit

    The multi-slit interference factor is [sin(Nγ)/sin(γ)]².
    """
    a_eff = d / 2.0            # effective slit width assumption
    beta  = np.pi * a_eff * x / (lam * D)
    gamma = np.pi * d     * x / (lam * D)

    # Single-slit envelope  (avoid divide-by-zero with np.where)
    sinc_part = np.where(np.abs(beta) < 1e-12,
                         1.0,
                         np.sin(beta) / beta)

    # Multi-slit interference
    num   = np.sin(N * gamma)
    denom = np.sin(gamma)
    # At grating maxima denom→0 and num→0; limit = N
    multi = np.where(np.abs(denom) < 1e-12,
                     float(N),
                     num / denom)

    # Normalise so peak = 1
    I = (sinc_part ** 2) * (multi ** 2)
    peak = float(N) ** 2
    return I / peak


# ─────────────────────────────────────────────────────────────────────────────
# Main application class
# ─────────────────────────────────────────────────────────────────────────────

class WaveOpticsApp:
    # ── colours ───────────────────────────────────────────────────────────────
    BG        = "#0d1117"
    PANEL_BG  = "#161b22"
    ACCENT    = "#58a6ff"
    TEXT      = "#e6edf3"
    SUBTEXT   = "#8b949e"
    BORDER    = "#30363d"
    BTN_ACT   = "#1f6feb"
    BTN_INACT = "#21262d"

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Wave Optics Simulator")
        self.root.configure(bg=self.BG)
        self.root.geometry("1280x820")
        self.root.minsize(1100, 700)

        # Active experiment: "double" | "single" | "grating"
        self.mode = tk.StringVar(value="double")
        self.diff_sub = tk.StringVar(value="single")   # sub-mode for diffraction

        self._build_layout()
        self._build_plot_area()          # axes must exist before sidebars call _update_plot
        self._build_sidebar_double()
        self._build_sidebar_diffraction()

        self._show_mode("double")   # start on double slit

    # ── layout skeleton ───────────────────────────────────────────────────────

    def _build_layout(self):
        """Three columns: nav | sidebar | plot."""
        self.root.columnconfigure(0, weight=0, minsize=130)
        self.root.columnconfigure(1, weight=0, minsize=240)
        self.root.columnconfigure(2, weight=1)
        self.root.rowconfigure(0, weight=1)

        # ── navigation bar (leftmost) ──────────────────────────────────────
        nav = tk.Frame(self.root, bg=self.PANEL_BG, width=130,
                       highlightbackground=self.BORDER, highlightthickness=1)
        nav.grid(row=0, column=0, sticky="nsew")
        nav.grid_propagate(False)

        # Logo / title
        tk.Label(nav, text="⬡", font=("Courier", 28), fg=self.ACCENT,
                 bg=self.PANEL_BG).pack(pady=(24, 2))
        tk.Label(nav, text="Wave\nOptics", font=("Courier", 10, "bold"),
                 fg=self.TEXT, bg=self.PANEL_BG, justify="center").pack()

        ttk.Separator(nav, orient="horizontal").pack(fill="x", padx=10, pady=16)

        self._nav_btn_double = self._nav_button(nav, "Double\nSlit",
                                                 lambda: self._show_mode("double"))
        self._nav_btn_diff   = self._nav_button(nav, "Diffraction",
                                                 lambda: self._show_mode("diff"))

        # ── sidebar ────────────────────────────────────────────────────────
        self.sidebar = tk.Frame(self.root, bg=self.PANEL_BG,
                                highlightbackground=self.BORDER,
                                highlightthickness=1)
        self.sidebar.grid(row=0, column=1, sticky="nsew")

        # ── plot area ──────────────────────────────────────────────────────
        self.plot_frame = tk.Frame(self.root, bg=self.BG)
        self.plot_frame.grid(row=0, column=2, sticky="nsew", padx=4, pady=4)

    def _nav_button(self, parent, text, cmd):
        btn = tk.Button(parent, text=text, command=cmd,
                        bg=self.BTN_INACT, fg=self.TEXT,
                        activebackground=self.BTN_ACT, activeforeground="white",
                        font=("Courier", 9), relief="flat", bd=0,
                        cursor="hand2", width=12, height=3, wraplength=100)
        btn.pack(pady=4, padx=8)
        return btn

    # ── sidebar panels ────────────────────────────────────────────────────────

    def _label(self, parent, text, sub=False):
        fnt = ("Courier", 8) if sub else ("Courier", 9, "bold")
        fg  = self.SUBTEXT if sub else self.TEXT
        tk.Label(parent, text=text, font=fnt, fg=fg,
                 bg=self.PANEL_BG, anchor="w").pack(fill="x", padx=14, pady=(8,0))

    def _slider(self, parent, var, from_, to, resolution, fmt=None):
        """Return a slider tied to *var*; updates plot on change."""
        frm = tk.Frame(parent, bg=self.PANEL_BG)
        frm.pack(fill="x", padx=14, pady=2)

        val_lbl = tk.Label(frm, textvariable=var, width=7,
                            font=("Courier", 8), fg=self.ACCENT, bg=self.PANEL_BG,
                            anchor="e")
        val_lbl.pack(side="right")

        s = tk.Scale(frm, variable=var, from_=from_, to=to,
                     resolution=resolution, orient="horizontal",
                     bg=self.PANEL_BG, fg=self.TEXT,
                     troughcolor=self.BORDER, activebackground=self.ACCENT,
                     highlightthickness=0, bd=0, sliderrelief="flat",
                     command=lambda _: self._update_plot())
        s.pack(fill="x")
        return s

    def _build_sidebar_double(self):
        """Sidebar panel for Young's double-slit experiment."""
        self._panel_double = tk.Frame(self.sidebar, bg=self.PANEL_BG)

        tk.Label(self._panel_double, text="Young's Double Slit",
                 font=("Courier", 11, "bold"), fg=self.ACCENT,
                 bg=self.PANEL_BG).pack(pady=(20, 4), padx=14, anchor="w")
        ttk.Separator(self._panel_double).pack(fill="x", padx=10, pady=6)

        # Tkinter variables (store values in display units)
        self.ds_lam = tk.DoubleVar(value=550)    # nm
        self.ds_d   = tk.DoubleVar(value=0.5)    # mm
        self.ds_a   = tk.DoubleVar(value=80)     # µm
        self.ds_D   = tk.DoubleVar(value=1.5)    # m

        self._label(self._panel_double, "Wavelength  λ  (nm)")
        self._slider(self._panel_double, self.ds_lam, 380, 780, 1)

        self._label(self._panel_double, "Slit separation  d  (mm)")
        self._slider(self._panel_double, self.ds_d, 0.1, 3.0, 0.01)

        self._label(self._panel_double, "Slit width  a  (µm)")
        self._slider(self._panel_double, self.ds_a, 10, 300, 1)

        self._label(self._panel_double, "Screen distance  D  (m)")
        self._slider(self._panel_double, self.ds_D, 0.5, 5.0, 0.05)

        self._label(self._panel_double,
                    "Fringes scale with λD/d.\nEnvelope widens as a↓.",
                    sub=True)

    def _build_sidebar_diffraction(self):
        """Sidebar panel for Single-slit / Grating diffraction."""
        self._panel_diff = tk.Frame(self.sidebar, bg=self.PANEL_BG)

        tk.Label(self._panel_diff, text="Diffraction",
                 font=("Courier", 11, "bold"), fg=self.ACCENT,
                 bg=self.PANEL_BG).pack(pady=(20, 4), padx=14, anchor="w")
        ttk.Separator(self._panel_diff).pack(fill="x", padx=10, pady=4)

        # Sub-mode toggle
        tog_frm = tk.Frame(self._panel_diff, bg=self.PANEL_BG)
        tog_frm.pack(fill="x", padx=14, pady=4)
        for txt, val in [("Single Slit", "single"), ("Grating", "grating")]:
            rb = tk.Radiobutton(tog_frm, text=txt, variable=self.diff_sub,
                                value=val, command=self._on_diff_sub_change,
                                bg=self.PANEL_BG, fg=self.TEXT,
                                selectcolor=self.BTN_ACT, activebackground=self.PANEL_BG,
                                font=("Courier", 9), indicatoron=True)
            rb.pack(side="left", padx=(0, 12))

        ttk.Separator(self._panel_diff).pack(fill="x", padx=10, pady=4)

        # Shared parameters
        self.df_lam = tk.DoubleVar(value=550)    # nm
        self.df_D   = tk.DoubleVar(value=2.0)    # m

        self._label(self._panel_diff, "Wavelength  λ  (nm)")
        self._slider(self._panel_diff, self.df_lam, 380, 780, 1)

        self._label(self._panel_diff, "Screen distance  D  (m)")
        self._slider(self._panel_diff, self.df_D, 0.5, 5.0, 0.05)

        ttk.Separator(self._panel_diff).pack(fill="x", padx=10, pady=8)

        # ── single-slit sub-panel ──────────────────────────────────────────
        self._sub_single = tk.Frame(self._panel_diff, bg=self.PANEL_BG)
        self.ss_a = tk.DoubleVar(value=100)      # µm
        self._label(self._sub_single, "Slit width  a  (µm)")
        self._slider(self._sub_single, self.ss_a, 10, 500, 1)
        self._label(self._sub_single,
                    "Minima at x = mλD/a  (m≠0).", sub=True)

        # ── grating sub-panel ──────────────────────────────────────────────
        self._sub_grating = tk.Frame(self._panel_diff, bg=self.PANEL_BG)
        self.gr_N  = tk.IntVar(value=5)          # number of slits
        self.gr_lpm = tk.DoubleVar(value=300)    # lines per mm  →  d = 1/lpm (mm)
        self._label(self._sub_grating, "Number of slits  N")
        self._slider(self._sub_grating, self.gr_N, 2, 20, 1)
        self._label(self._sub_grating, "Lines per mm")
        self._slider(self._sub_grating, self.gr_lpm, 50, 1200, 10)
        self._label(self._sub_grating,
                    "Maxima at d·sinθ = mλ.\nMore slits → sharper peaks.",
                    sub=True)

        self._on_diff_sub_change()   # show correct sub-panel initially

    def _on_diff_sub_change(self):
        """Switch between single-slit and grating sub-panels."""
        self._sub_single.pack_forget()
        self._sub_grating.pack_forget()
        if self.diff_sub.get() == "single":
            self._sub_single.pack(fill="x")
        else:
            self._sub_grating.pack(fill="x")
        self._update_plot()

    # ── plot area ─────────────────────────────────────────────────────────────

    def _build_plot_area(self):
        """Create the matplotlib Figure embedded in the Tk window."""
        self.fig = Figure(figsize=(8, 6), dpi=100,
                          facecolor=self.BG, tight_layout=True)

        # Two rows: intensity profile (top) + screen view (bottom)
        gs = self.fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.35)
        self.ax_intensity = self.fig.add_subplot(gs[0])
        self.ax_screen    = self.fig.add_subplot(gs[1])

        self._style_axes()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def _style_axes(self):
        for ax in (self.ax_intensity, self.ax_screen):
            ax.set_facecolor("#0a0e14")
            ax.tick_params(colors=self.SUBTEXT, labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor(self.BORDER)
        self.ax_intensity.set_xlabel("Position  x  (mm)",
                                      color=self.SUBTEXT, fontsize=9)
        self.ax_intensity.set_ylabel("Normalised Intensity",
                                      color=self.SUBTEXT, fontsize=9)
        self.ax_screen.set_yticks([])
        self.ax_screen.set_xlabel("Position  x  (mm)",
                                   color=self.SUBTEXT, fontsize=9)

    # ── mode switching ────────────────────────────────────────────────────────

    def _show_mode(self, mode: str):
        """Switch sidebar panel and highlight the active nav button."""
        self._panel_double.pack_forget()
        self._panel_diff.pack_forget()

        if mode == "double":
            self._panel_double.pack(fill="both", expand=True)
            self._nav_btn_double.config(bg=self.BTN_ACT)
            self._nav_btn_diff.config(bg=self.BTN_INACT)
            self.mode.set("double")
        else:
            self._panel_diff.pack(fill="both", expand=True)
            self._nav_btn_double.config(bg=self.BTN_INACT)
            self._nav_btn_diff.config(bg=self.BTN_ACT)
            self.mode.set("diff")

        self._update_plot()

    # ── plot update ───────────────────────────────────────────────────────────

    def _update_plot(self, *_):
        """Recompute intensity and refresh both axes."""
        # Guard: axes may not exist yet during __init__ sequencing
        if not hasattr(self, "ax_intensity"):
            return
        self.ax_intensity.cla()
        self.ax_screen.cla()
        self._style_axes()

        # ── compute ────────────────────────────────────────────────────────
        if self.mode.get() == "double":
            lam = self.ds_lam.get() * 1e-9      # nm  → m
            d   = self.ds_d.get()   * 1e-3      # mm  → m
            a   = self.ds_a.get()   * 1e-6      # µm  → m
            D   = self.ds_D.get()               # m

            # Screen half-width: show ≈4 central fringes
            half_x = 4 * lam * D / d
            x = np.linspace(-half_x, half_x, 4000)
            I = intensity_double_slit(x, lam, d, a, D)
            title = f"Young's Double Slit  |  λ={self.ds_lam.get():.0f} nm  " \
                    f"d={self.ds_d.get():.2f} mm  a={self.ds_a.get():.0f} µm"
            wl_nm = self.ds_lam.get()

        else:  # diffraction
            lam = self.df_lam.get() * 1e-9
            D   = self.df_D.get()
            wl_nm = self.df_lam.get()

            if self.diff_sub.get() == "single":
                a = self.ss_a.get() * 1e-6     # µm → m
                half_x = 5 * lam * D / a
                x = np.linspace(-half_x, half_x, 4000)
                I = intensity_single_slit(x, lam, a, D)
                title = f"Single-Slit Diffraction  |  λ={wl_nm:.0f} nm  " \
                        f"a={self.ss_a.get():.0f} µm"
            else:
                N  = int(self.gr_N.get())
                lpm = self.gr_lpm.get()          # lines per mm
                d_gr = 1.0 / (lpm * 1e3)         # grating element in metres
                # Show 3 orders on each side
                half_x = 3 * lam * D / d_gr
                half_x = min(half_x, D * 0.5)   # physical limit
                x = np.linspace(-half_x, half_x, 6000)
                I = intensity_grating(x, lam, d_gr, N, D)
                title = f"Diffraction Grating  |  λ={wl_nm:.0f} nm  " \
                        f"N={N}  {lpm:.0f} l/mm"

        x_mm = x * 1e3      # convert to mm for display

        # ── intensity profile ──────────────────────────────────────────────
        rgb = wavelength_to_rgb(wl_nm)
        self.ax_intensity.plot(x_mm, I, color=rgb, linewidth=1.1, alpha=0.95)
        self.ax_intensity.fill_between(x_mm, I, alpha=0.18, color=rgb)
        self.ax_intensity.set_xlim(x_mm[0], x_mm[-1])
        self.ax_intensity.set_ylim(-0.04, 1.1)
        self.ax_intensity.set_title(title, color=self.TEXT,
                                     fontsize=9, pad=6)
        self.ax_intensity.axhline(0, color=self.BORDER, linewidth=0.6)

        # ── screen (imshow) ───────────────────────────────────────────────
        # Build a 2-D image: stack the 1-D intensity row vertically
        screen_height = 60
        screen_img = np.tile(I[np.newaxis, :], (screen_height, 1))
        cmap = make_screen_colormap(wl_nm)
        self.ax_screen.imshow(screen_img, aspect="auto",
                              extent=[x_mm[0], x_mm[-1], 0, 1],
                              cmap=cmap, vmin=0, vmax=1,
                              origin="lower")
        self.ax_screen.set_xlim(x_mm[0], x_mm[-1])
        self.ax_screen.set_title("Simulated Screen",
                                  color=self.SUBTEXT, fontsize=8, pad=4)

        self.canvas.draw_idle()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    root = tk.Tk()
    app = WaveOpticsApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()