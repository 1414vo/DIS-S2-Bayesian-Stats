import matplotlib.pyplot as plt


def trace_plot(chain, param_names, title=None, out_path=None):
    """! Displays the trace plot for a generated Markov chain.

    @param chain        The generated Markov Chain
    @param param_names  The names of the parameters.
    @param title        Customizable plot title.
    @param out_path     The output path of where to save the plot."""
    plt.figure(figsuize=(16, 7))
    for i, param in enumerate(param_names):
        plt.plot(chain[:, i], label=param, alpha=0.35)

    plt.xlabel("Chain index")
    plt.ylabel("Parameter Value")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if out_path is None:
        plt.show()
    else:
        plt.savefig(out_path, dpi=300)


def autocorr_plot(chain, param_names, title=None, out_path=None):
    """! Displays the autocorrelation plots for a generated Markov chain.

    @param chain        The generated Markov Chain
    @param param_names  The names of the parameters.
    @param title        Customizable plot title.
    @param out_path     The output path of where to save the plot."""
    plt.figure(figsuize=(16, 5 * len(param_names) + 2))
    for i, param in enumerate(param_names):
        # Generate autocorrelation
        plt.subplot(i + 1, 1, len(param_names))
        plt.acorr(chain[:, i], label=param, max_lags=200)

    plt.xlabel("Series lag")
    plt.text(0.04, 0.5, "Autocorrelation coefficient", va="center", rotation="vertical")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if out_path is None:
        plt.show()
    else:
        plt.savefig(out_path, dpi=300)
