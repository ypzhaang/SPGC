# Multilayer Similarity-Preserving Graph Coarsening Framework for Knowledge Tree Constructionn

## Overview
>   We conduct a study on the problem of Knowledge Tree Construction (KoTC), which aims to organize knowledge-concept (KC) nodes from a graph into a cognitively hierarchical structure, enhancing the graph learning and understanding. Given a graph, graph coarsening (GC) extracts a smaller graph while preserving the properties of the original graph. Towards KoTC, we propose the multilayer similarity-preserving graph coarsening (Multilayer SPGC) framework. Specifically, SPGC preserves the similarity in GC to encourage similar nodes to be aggregated into high-order nodes, while the  multilayer version performs SPGC many times to achieve coarsened graphs at the multiple levels of KoT. The proposed method is formulated into a optimization problem that can be efficiently solved by the block Majorization-Minimization algorithm. Using Multilayer SPGC, we can construct a KoT from any given graph and then employ graph neural networks to learn features at the levels of the KoT, followed by integrating various features into a multilayer perceptron for prediction. Experiment results on building course tree show that Multilayer SPGC can capture semantic similarity so that the generated KoT is aligned with human priors, while the results for graph classification show that Multilayer SPGC achieves the better performance than the state-of-the-art GC methods. This study provides a solution to not only graph learning but also KC graph construction.

## Multilayer SPGC
![flow chart](pictures/flow chart.png)
## Dependencies
>The code requires Python >= 3.9 and PyTorch >= 1.10.1.
>More details of the environment dependencies required for code execution can be found in the `requirements.txt` file within the repository.



 
