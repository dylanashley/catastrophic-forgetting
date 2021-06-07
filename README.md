# catastrophic-forgetting

This repository contains the source code for the experiments presented in *Does the Adam Optimizer Exacerbate Catastrophic Forgetting?* by Dylan R. Ashley, Sina Ghiassian, and Richard S. Sutton.

The intention of providing this code is to ensure transparency and replicability with regards to the results presented in the paper. Note that running these experiments and generating the relevant visualizations is computationally expensive and somewhat involved. Access to a supercomputer is recommended.

To generate the data presented in the paper, a user should first ensure they have `Python 3.7.4` installed with `NumPy 1.19.2`, `Pandas 1.1.3`, `TensorFlow 2.3.0`, `PyTorch 1.7.1`, and `OpenAI Gym 0.18.0`. Assuming they have access to a supercomputer with the Slurm Workload Manager, they should then examine the `run.sh` file. Each of the four blocks contained in the `run.sh` should be executed in sequence from the root directory of this repository. A user without access to a supercomputer with Slurm will need to adapt the `run.sh` file to fit their specific computational resources.

The `results.ipynb` notebook can be used to visualize the resulting data. To use this notebook, users will additionally need `Jupyter 6.0.2` installed with `Matplotlib 3.1.1`, `SciPy 1.4.1`, and `Seaborn 0.11.1`. The `results.ipynb` file contains multiple independent sections. To generate the plots, execute the `Setup` section of the notebook first, and then the section corresponding to the selected testbed and section.

Any questions or requests for assistance in operating this code can be directed to [Dylan R. Ashley](https://dylanashley.ca/).
