import matplotlib.pyplot as plt

PALETTE = {
    "Black": "#000000",
    "Orange": "#E69F00",
    "Sky Blue": "#56B4E9",
    "Bluish Green": "#009E73",
    "Yellow": "#F0E442",
    "Blue": "#0072B2",
    "Vermillion": "#D55E00",
    "Reddish Purple": "#CC79A7",
}


def hex_to_rgba(hex_color, alpha=1.0):
    """Convert hex color to RGBA."""
    hex_color = hex_color.lstrip("#")
    lv = len(hex_color)
    return tuple(
        int(hex_color[i : i + lv // 3], 16) / 255.0 for i in range(0, lv, lv // 3)
    ) + (alpha,)


def get_color(color_name, alpha=1.0):
    """Return the RGBA code of the given color name with specified transparency."""
    hex_color = PALETTE.get(
        color_name, "#000000"
    )  # Default to black if color not found
    return hex_to_rgba(hex_color, alpha)


def get_color_list():
    """Return a list of color hex codes from the palette."""
    return list(PALETTE.values())


def visualize_palette():
    """Visualize the color palette."""
    fig, ax = plt.subplots(figsize=(20, 2))
    for i, (color_name, color_hex) in enumerate(PALETTE.items()):
        ax.add_patch(plt.Rectangle((i * 2, 0), 2, 2, color=color_hex))
        ax.text(
            i * 2 + 1,
            1,
            color_name,
            color="black" if color_name == "White" else "white",
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
        )
    ax.set_xlim(0, 2 * len(PALETTE))
    ax.set_ylim(0, 2)
    ax.axis("off")
    plt.show()


####################

###  ROLAND's cmds for making pdf_tex plots work
# mpl.rcParams["font.family"] = "Open Sans"
# mpl.rcParams.update({"font.size": 14})
# mpl.rcParams["figure.figsize"] = 10, 5.625
# mpl.rc("xtick", labelsize=13)
# mpl.rc("ytick", labelsize=13)
# mpl.rcParams["pdf.fonttype"] = 42  # Output Type 3 (Type3) or Type 42(TrueType)

# # disable outline paths for inkscape > PDF+Latex
# # important: comment out all other local font settings
# mpl.rcParams["svg.fonttype"] = "none"

# # Path to the directory where your fonts are installed
# font_dir = "/home/jellepoland/.local/share/fonts/"

# # Add each font in the directory
# for font_file in os.listdir(font_dir):
#     if font_file.endswith(".ttf") or font_file.endswith(".otf"):
#         fm.fontManager.addfont(os.path.join(font_dir, font_file))

# def saving__pdf_tex(results_dir: str, filename: str):
# plt.savefig(Path(results_dir) / "pdf_tex" / f"{filename}.pdf")
# plt.savefig(Path(results_dir) / "pdf_tex" / f"{filename}.pdf_tex", format="pgf")
# plt.close()

#####################


def set_plot_style():
    """
    Set the default style for plots using LaTeX and custom color palette.

    Tips:
    - If you specify colors, they will still be used.
    - If you want to change the axis margins:
        1. try with ax.xlim and ax.ylim
        2. try by changing the 'axes.autolimit_mode' parameter to data
    - more?
    """
    import matplotlib.pyplot as plt
    from cycler import cycler

    # Define the color palette as a list of colors
    color_cycle = [
        PALETTE["Black"],
        PALETTE["Orange"],
        PALETTE["Sky Blue"],
        PALETTE["Bluish Green"],
        PALETTE["Yellow"],
        PALETTE["Blue"],
        PALETTE["Vermillion"],
        PALETTE["Reddish Purple"],
    ]

    # Apply Seaborn style and custom settings
    # plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            ## Axes settings
            "axes.titlesize": 15,
            "axes.labelsize": 15,
            "axes.linewidth": 1.0,
            "axes.edgecolor": "#C5C5C5",
            "axes.labelcolor": "black",
            "axes.autolimit_mode": "round_numbers",
            "axes.xmargin": 0,  # Remove extra margin
            "axes.ymargin": 0,  # Remove extra margin
            "axes.grid": True,  # Add gridlines by default
            ## Grid settings
            "axes.grid": True,
            "axes.grid.axis": "both",
            "grid.alpha": 0.5,
            "grid.color": "#C5C5C5",
            "grid.linestyle": "-",
            "grid.linewidth": 1.0,
            ## Line settings
            "lines.linewidth": 1,
            "lines.markersize": 6,
            # "lines.color": "grey",,
            "figure.titlesize": 15,
            "pgf.texsystem": "pdflatex",  # Use pdflatex
            "pgf.rcfonts": False,
            "figure.figsize": (15, 5),  # Default figure size
            "axes.prop_cycle": cycler(
                "color", color_cycle
            ),  # Set the custom color cycle
            ## tick settings
            "xtick.color": "#C5C5C5",
            "ytick.color": "#C5C5C5",
            "xtick.labelcolor": "black",
            "ytick.labelcolor": "black",
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "xtick.top": True,  # Show ticks on both sides
            "xtick.bottom": True,
            "ytick.left": True,
            "ytick.right": True,
            "xtick.direction": "in",  # Direction for x-axis ticks
            "ytick.direction": "in",  # Direction for y-axis ticks
            ## legend settings
            "legend.fontsize": 15,
        }
    )


def set_plot_style_no_latex():
    """Set the default style for plots without requiring LaTeX."""
    import matplotlib.pyplot as plt
    from cycler import cycler

    # Define the color palette as a list of colors
    color_cycle = [
        PALETTE["Black"],
        PALETTE["Orange"],
        PALETTE["Sky Blue"],
        PALETTE["Bluish Green"],
        PALETTE["Yellow"],
        PALETTE["Blue"],
        PALETTE["Vermillion"],
        PALETTE["Reddish Purple"],
    ]

    # Apply Seaborn style and custom settings
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "text.usetex": False,  # Disable LaTeX
            "font.family": "serif",
            "font.serif": [
                "DejaVu Serif"
            ],  # Use a serif font that is similar to LaTeX’s default font
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "axes.linewidth": 1,
            "lines.linewidth": 1,
            "lines.markersize": 6,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 16,
            "figure.figsize": (10, 6),  # Default figure size
            "axes.prop_cycle": cycler(
                "color", color_cycle
            ),  # Set the custom color cycle
        }
    )


# Optionally, you can also include a function to apply the palette to a plot
def apply_palette(ax, colors):
    """Apply the color palette to a matplotlib axis."""
    for line, color in zip(ax.get_lines(), colors):
        line.set_color(color)
    plt.draw()


def plot_on_ax(
    ax,
    x,
    y,
    label: str,
    color: str = None,
    linestyle: str = "-",
    marker: str = None,
    markersize: int = None,
    is_with_grid: bool = True,
    is_return_ax: bool = False,
    x_label: str = "X-axis",
    y_label: str = "Y-axis",
    is_with_x_label: bool = True,
    is_with_y_label: bool = True,
    is_with_x_ticks: bool = True,
    is_with_y_ticks: bool = True,
    title: str = None,
):
    """Plot data on a given axis."""

    # turning off the ticks
    if not is_with_x_ticks:
        ax.tick_params(labelbottom=False)
    if not is_with_y_ticks:
        ax.tick_params(left=False, right=False, labelleft=False)

    if is_with_grid:
        ax.tick_params(
            bottom=False,
            top=False,
            left=False,
            right=False,
        )
    else:
        ax.grid(False)

    if color is None:
        if marker is None:
            if markersize is None:
                ax.plot(x, y, label=label, linestyle=linestyle)
            else:
                ax.plot(x, y, label=label, linestyle=linestyle, markersize=markersize)
        else:
            ax.plot(x, y, label=label, linestyle=linestyle, marker=marker)
    else:
        if marker is None:
            ax.plot(x, y, label=label, color=color, linestyle=linestyle)
        else:
            if markersize is None:
                ax.plot(
                    x,
                    y,
                    label=label,
                    color=color,
                    linestyle=linestyle,
                    marker=marker,
                )
            else:
                ax.plot(
                    x,
                    y,
                    label=label,
                    color=color,
                    linestyle=linestyle,
                    marker=marker,
                    markersize=markersize,
                )
    if title is not None:
        ax.set_title(title)

    if is_with_x_label:
        ax.set_xlabel(x_label)
    if is_with_y_label:
        ax.set_ylabel(y_label)
    if is_return_ax:
        return ax
