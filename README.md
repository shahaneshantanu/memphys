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

## Cite
If you use this software, please cite the following [research paper](https://arxiv.org/abs/2010.01702)<br/>
Shahane, S., Radhakrishnan, A., & Vanka, S. P. (2020). A High-Order Accurate Meshless Method for Solution of Incompressible Fluid Flow Problems. arXiv preprint arXiv:2010.01702.<br/>
BibTeX entry:<br/>
@article{shahane2020high,
  title={A High-Order Accurate Meshless Method for Solution of Incompressible Fluid Flow Problems},
  author={Shahane, Shantanu and Radhakrishnan, Anand and Vanka, Surya Pratap},
  journal={arXiv preprint arXiv:2010.01702},
  year={2020}
}

## Credits:
Dr. Shantanu Shahane, Postdoctoral Research Associate<br/>
Dr. Surya Pratap Vanka, Professor Emeritus<br/>
Mechanical Science and Engineering, University of Illinois at Urbana-Champaign

## License:
MeMPhyS is published under [MIT License](https://github.com/shahaneshantanu/memphys/blob/main/LICENSE)
