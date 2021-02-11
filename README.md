# catastrophic-forgetting

This repository contains the source code for the experiments presented in *Does Standard Backpropagation Forget Less Catastrophically Than Adam?* by Dylan R. Ashley, Sina Ghiassian, and Richard S. Sutton.

The intention of providing this code is to ensure transparency and replicability with regards to the results presented in the paper. Note that running these experiments and generating the relevant visualizations is computationally expensive and somewhat involved. Access to a supercomputer is recommended.

To generate the data presented in the paper, a user should first ensure they have Python 3.7.4 installed with NumPy 1.17.4, Pandas 0.25.3, Tensorflow 2.2.0, Pytorch 1.5.1, and OpenAI Gym 0.17.1. They should then examine the `build.sh` file appearing in each of the three directories corresponding to each of the three settings appearing in the thesis. Each of the `build.sh` files contains a section for the alpha selection procedure (Train) and a section for generating the final results under the selected alpha (Test). Only one section for a given testbed can be run at a time, so comment out sections until exactly one section isn't commented out.

Once you've finished commenting out sections in one of the `build.sh` files, execute the file. Doing so will generate a number of `tasks_*.sh` files. Execute each of these files to generate the data corresponding to the selected testbed and section. Note that these files contain a series of independent experiments to run. As such, each file--and further each line in each file--can be executed independently and in parallel. After all the files have completed execution, run the `merge.sh` file to conglomerate the data.

The `results.ipynb` notebook can be used to visualize the resulting data. To use this notebook, users will additionally need Jupyter 1.0.0 installed with Matplotlib 3.1.0, SciPy 1.3.0, and Seaborn 0.9.0. As with the `build.sh` file, the `results.ipynb` file contains multiple sections. To generate the plots, ensure the data is named as the notebook expects and execute the `Setup` section of the notebook first, and then the section corresponding to the selected testbed and section.

Any questions or requests for assistance in operating this code can be directed to [Dylan R. Ashley](https://dylanashley.ca/).
