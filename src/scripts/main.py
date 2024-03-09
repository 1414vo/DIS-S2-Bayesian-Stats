import argparse
from .part_v import execute_part_v
from .part_vii import execute_part_vii

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Lighthouse Parameter Estimation",
        description="Estimates the lighthouse location parameters using different MCMC algorithms.",
    )
    parser.add_argument("data_path", help="Location of the data file.")
    parser.add_argument("out_path", help="Location of the output folder.")

    args = parser.parse_args()

    # Execute part V script
    print("PARAMETER ESTIMATION FROM DETECTION LOCATIONS")
    print("===============================================================")
    execute_part_v(args.data_path, args.out_path)

    # Execute part VII script
    print("\nPARAMETER ESTIMATION FROM DETECTION LOCATIONS AND INTENSITIES")
    print("===============================================================")
    execute_part_vii(args.data_path, args.out_path)
