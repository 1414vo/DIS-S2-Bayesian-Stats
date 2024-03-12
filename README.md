# S2_Coursework - Ivo Petrov
![Static Badge](https://img.shields.io/badge/build-passing-lime)
![Static Badge](https://img.shields.io/badge/logo-gitlab-blue?logo=gitlab)

## Table of contents
1. [Description](#description)
2. [Installation](#Installation)
3. [Usage](#usage)
4. [Auto-documentation](#auto-documentation)
5. [Major Dependencies](#major-dependencies)
6. [License](#license)
7. [Credits](#credits)

## Description
This project contains an approach of Bayesian inference applied to the Lighthouse problem. All the relevant code is contained in the `/src` folder.
The major features of the project include:
- 3 different sampling techniques (`sampling.py`):
    - Metropolis-Hastings (hand-coded)
    - Nested Sampling using `nessai`
    - Ensemble Sampling using `emcee`
- Relevant plotting utilities (`plotting.py`):
    - Trace plots
    - Autocorrelation plots
    - Corner plots
    - Distribution comparison
- Diagnostic measurements (`diagnostics.py`):
    - Simple summary
    - Gelman-Rubin statistic
    - Univariate Kolmogorov-Smirnov test
    - Kullback-Liebler divergence estimation

Alongside the code, following best practices, we include:
- Unit tests for checking certain components of the code work correctly.
- Pre-commit setup to ensure the code conforms to the PEP8 style.
- Environment file containing all dependencies.
- Doxygen for auto-documentation.

## Installation
We provide 2 ways of installing the relevant dependencies and setting up the environment correctly - through Docker or Conda. Docker is preferred due to its high reliability across different operating systems and hardware. However, Conda is still a reliable enough option that requires less effort to set up.

### Using Docker
1. With the Dockerfile provided, create an image using:

```docker build -t s2_ivp24 .```

2. Run the image using:

```docker run --name <name> --rm -ti s2_ivp24```

3. (Optional) If you want to change something in the Git repository, docker does not automatically infer your credentials, so you can mount them using the following script when:

```docker run --name <name> --rm -v <ssh folder on local machine>:/root/.ssh -ti s2_ivp24```

This copies your keys to the created container and you should be able to run all required git commands.

### Using Conda

Setting up the Conda environment can be done using the following commands:

``conda create -n s2_ivp24 python``

``conda env update --file environment.yml --name s2_ivp24``

``conda activate s2_ivp24``

## Usage
The preferred way to run the script is:

``python -m src.scripts.main <data_path> <output_path>``

If you would like to replicate the Kullback-Liebler Divergence measurements, as shown in the project, please enable the `--kld` tag, as follows:

``python -m src.scripts.main <data_path> <output_path> --kld``
This was selected as an opt-in option due to the high computation costs. It should be noted that the full script takes around **60 - 80 minutes** to complete, while the one without KLD runs for between **15-20 minutes**. This is mostly due to the inefficiency in the hand-coded Metropolis-Hastings algorithm.

The relevant plots can be found in the specified output path, with the numerical results being displayed in the command line. If the way you are running the script through a Docker container terminal, you can copy them to your local machine via the following script:

``docker cp s2:/S2_Coursework/<output_path> ./out``

Finally, if you wish to run either Part (v) or Part (vii) separately, use one of the following scripts:

``python -m src.scripts.part_v <data_path> <output_path> <--kld>``

or

``python -m src.scripts.part_vii <data_path> <output_path> <--kld>``

## Auto-documentation

If setup correctly, you should be able to generate the auto-documentation by running the `doxygen` command. The relevant files will be built inside the `/docs` folder. We recommend using the `html` version, as the $\LaTeX$ is often unstable. To do that, simply open the `index.html` file in your browser. You might need to run this script to transfer the files to your local machine:

``docker cp s2:/S2_Coursework/docs/html ./docs``

## Major Dependencies
A list of all dependencies can be found in the `environment.yml` file. However, the core packages used in the project can be found below:

- numpy
- scipy
- matplotlib
- arviz
- nessai
- emcee

## License
This project is licensed under the GNU General Public License v3.0.

### Permissions and Limitations:
-----------------------------
- Users are allowed to modify and distribute this software.
- Users must provide the source code when distributing this software.
- Users can use this software for both commercial and non-commercial purposes.

## Credits
Repository was created using a script provided by James Fergusson towards the DIS Research Computing course.

The core of the project was done thanks to the development of state of the art sampling models, namely Michael J. Williams's [`nessai`](https://nessai.readthedocs.io/en/latest/) and Dan Foreman-Mackey et al.'s [`emcee`]. The relevant papers can be found at:
 - `nessai`: https://arxiv.org/abs/2302.08526
 - `emcee` : https://arxiv.org/abs/1202.3665
