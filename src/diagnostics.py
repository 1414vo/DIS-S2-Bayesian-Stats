import arviz


def chain_convergence_diagnostics(chains, samples, param_names):
    """! Computes certain diagnostic statistics for the Markov chain and
    the extract samples"""
    # Create Arziv dataset
    dataset = {param: chains[:, :, i] for i, param in enumerate(param_names)}

    # Simple summary
    print(
        "Effective number of samples is {frac: .2f}% of the total samples".format(
            frac=len(samples) / chains.shape[0] / chains.shape[1] * 100
        )
    )

    # Compute Gelman-Rubic Diagnostic (R hat)
    r_hats = arviz.diagnostics.rhat(dataset)
    for param in param_names:
        print(
            f'Measured Gelman-Rubin statistic for parameter "{param}": {r_hats[param]}'
        )
