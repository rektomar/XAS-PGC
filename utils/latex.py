from pylatex import Document, TikZ, Axis, Plot
from pylatex.utils import NoEscape


def parse_coords(x, y):
    return " ".join([f"({x_val},{y_val})" for x_val, y_val in zip(x, y)])

def create_line_plot(x, y, label, line_width=0.7):
    coords = parse_coords(x, y)
    plot_data = f"""
    \\addplot+[
        mark=none,
        line width={line_width}pt
    ] coordinates {{
        {coords}
    }};
    \\addlegendentry{{ {label} }}
    """
    return plot_data

def create_conf_line_plot(x, y_mean, y_std, label, line_width=0.7):
    mean_coords  = parse_coords(x, y_mean)
    upper_coords = parse_coords(x, y_mean + y_std)
    lower_coords = parse_coords(x, y_mean - y_std)
    plot_data = f"""
    \\addplot+[
        mark=none,
        name path=upper,
        opacity=0.2,
        forget plot
    ] coordinates {{
        {upper_coords}
    }};

    \\addplot+[
        mark=none,
        name path=lower,
        opacity=0.2,
        forget plot
    ] coordinates {{
        {lower_coords}

    }};
    \\addplot+[opacity=0.2, forget plot] fill between[of=upper and lower];

    \\addplot+[
        mark=none,
        line width={line_width}pt
    ] coordinates {{
        {mean_coords}
    }};
    \\addlegendentry{{ {label} }}
    """
    return plot_data

def create_latex_pgf_plot(plots: list, xlabel: str, ylabel: str, title:str, xlim=None, ylim=None, pdf_filename: str='pgf_plot'):
    doc = Document(documentclass='standalone', document_options=('preview'))

    doc.preamble.append(NoEscape(r'\usepackage{pgfplots}'))
    doc.preamble.append(NoEscape(r'\pgfplotsset{compat=newest}'))
    doc.preamble.append(NoEscape(r'\usepgfplotslibrary{fillbetween}'))

    doc.preamble.append(NoEscape(r'\usepgfplotslibrary{colorbrewer}'))
    # doc.preamble.append(NoEscape(r'\pgfplotsset{colormap/Dark2}'))

    if xlim is not None and ylim is not None:   
        xmin, xmax = xlim
        ymin, ymax = ylim
        doc.preamble.append(NoEscape(f'\pgfplotsset{{xmin={{{xmin}}}, xmax={{{xmax}}}, ymin={{{ymin}}}, ymax={{{ymax}}}}}'))

    with doc.create(TikZ()) as tikz:
        axis_options = NoEscape(
            f'width=12cm, height=8cm, xlabel={{{xlabel}}}, ylabel={{{ylabel}}}, title={{{title}}}, legend style={{at={{(1.05,1)}}, anchor=west}}, grid=major, cycle list/Dark2'
        )
        with tikz.create(Axis(options=axis_options)) as plot_axis:
            for plot_data in plots:
                # Add the plot data to the axis
                plot_axis.append(NoEscape(plot_data))

    doc.generate_pdf(pdf_filename, clean_tex=True)
    print(f"PDF document saved as {pdf_filename}")
