# MeMPhyS
# Meshless Multi-Physics Software
Details of the installation, setup, usage and examples can be found in the [Manual](https://github.com/shahaneshantanu/memphys/blob/main/Manual.pdf)

## Software Capabilities
MeMPhyS solves partial differential equations arising from various engineering and physics problems. It uses the Polyharmonic spline radial basis function with appended polynomials to estimate the differential operators over point clouds. Current capabilities include:
* Interpolation over scattered points
* Gradient and Laplacian stencils for 2D and 3D geometries
* Heat conduction (Poisson's equation) with Dirichlet and Neumann boundary conditions
* Scalar transport equation
* Incompressible fluid flow problems in 2D (Navier-Stokes equations)
* Forced convection problems
* Solidification of metal alloys for 2D and 3D geometries

## Examples
The [examples](https://github.com/shahaneshantanu/memphys/tree/main/examples) folder has following problems setup under various sub-folders:
* Problems involving single scalar variable: [advection_diffusion](https://github.com/shahaneshantanu/memphys/tree/main/examples/advection_diffusion)
* Two-dimensional fluid flow problems: [navier_stokes_2D](https://github.com/shahaneshantanu/memphys/tree/main/examples/navier_stokes_2D)
* Miscellaneous codes for analysis of stencils: [miscellaneous](https://github.com/shahaneshantanu/memphys/tree/main/examples/miscellaneous)
* Solidification problems: [solidification](https://github.com/shahaneshantanu/memphys/tree/main/examples/solidification)

## List of Publications
If you use MeMPhyS, please cite the following research papers:
1. N Bartwal, S Shahane, S Roy, SP Vanka (2021). Application of a High Order Accurate Meshless Method to Solution of Heat Conduction in Complex Geometries: [arXiv](https://arxiv.org/abs/2106.08535)<br/>
2. S Shahane, SP Vanka (2021). A Semi-Implicit Meshless Method for Incompressible Flows in Complex Geometries: [arXiv](https://arxiv.org/abs/2106.07616)<br/>
3. A Radhakrishnan, M Xu, S Shahane, SP Vanka (2021). A Non-Nested Multilevel Method for Meshless Solution of the Poisson Equation in Heat Transfer and Fluid Flow: [arXiv](https://arxiv.org/abs/2104.13758)<br/>
4. S Shahane, A Radhakrishnan, SP Vanka (2020). A High-Order Accurate Meshless Method for Solution of Incompressible Fluid Flow Problems: [arXiv](https://arxiv.org/abs/2010.01702)<br/>

## Credits:
Dr. Shantanu Shahane, Postdoctoral Research Associate (sshahan2@illinois.edu)<br/>
Dr. Surya Pratap Vanka, Professor Emeritus (spvanka@illinois.edu)<br/>
University of Illinois at Urbana-Champaign

## License:
MeMPhyS is published under [MIT License](https://github.com/shahaneshantanu/memphys/blob/main/LICENSE)
