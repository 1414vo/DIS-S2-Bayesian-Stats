import argparse
from .part_v import execute_part_v
from .part_vii import execute_part_vii
from ..plotting import plot_cauchy_convergence

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Lighthouse Parameter Estimation",
        description="Estimates the lighthouse location parameters using different MCMC algorithms.",
    )
    parser.add_argument("data_path", help="Location of the data file.")
    parser.add_argument("out_path", help="Location of the output folder.")
    parser.add_argument(
        "--kld",
        dest="kld",
        default=False,
        action="store_true",
        help="Whether to execute Kullback-Liebler Divergence estimations.",
    )
    args = parser.parse_args()

    # Demonstrate Cauchy mean non-convergence
    plot_cauchy_convergence(f"{args.out_path}/cauchy_convergence.png")

    # Execute part V script
    print("PARAMETER ESTIMATION FROM DETECTION LOCATIONS")
    print("===============================================================")
    execute_part_v(args.data_path, args.out_path, args.kld)

    # Execute part VII script
    print("\nPARAMETER ESTIMATION FROM DETECTION LOCATIONS AND INTENSITIES")
    print("===============================================================")
    execute_part_vii(args.data_path, args.out_path, args.kld)
