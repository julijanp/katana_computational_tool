# KATANA computational tool

Using Python to predict the activity inside the measurement snail in the KATANA water activation experiment. 
Reaction rates in irradiation volume calculated with MCNP.

The KATANA computational tool models the water activation loop using a conventional analytical approach, divided into four sections: the irradiation region (inner irradiation Snail), the observation region (outer Snail No. 1), and two transport regions (pipes and pump). The circuit is discretised into uniform volume elements of 21.65 cm $^3$, which are sequentially transported through the simulated loop. For each time step $\Delta t$, the specific activity of a given isotope in each volume element is updated according to


$A'_{t+\Delta t} = A'_t\ e^{-\lambda \Delta t} + R(1-e^{-\lambda \Delta t})$,


where $A'_t$ is the isotope-specific activity at time $t$, $\lambda$ is the decay constant, and $R$ is the average reaction rate within the region of interest. The latter is defined as  


$R = C \int \Phi(E)\,\sigma(E)\,N\,dE$.


Here, $\Phi(E)$ denotes the neutron flux, $\sigma(E)$ the microscopic cross section, and $N$ the atomic number density of the parent nuclide, calculated using the JSI TRIGA MCNP model [1]. The transport of each volume element proceeds in discrete steps determined by the volumetric flow rate.

![KATANA computational tool schematica](figures/25_10_02_katana_tool_scheme.png)

The computational tool is implemented in the Python programming language and can model a wide range of irradiation scenarios and loop geometries. Two main loop configurations are available: the short loop, optimized for radionuclides with short half-lives (N-16 and N-17), and the decay loop, which enables measurements of longer-lived activation products (O-19). In addition, the tool allows simulation of both flow transients and reactor power transients, providing flexibility for benchmarking experiments and validation studies under real experimental conditions.


Uncertainties in the KATANA computational tool are evaluated by combining contributions from nuclear data uncertainties, statistical uncertainties from Monte Carlo transport calculations, and the averaging of reaction rates across the irradiation volume. 

## References

[1] [D. Kotnik, J. Peric, D. Govekar, L. Snoj, and I. Lengar. “KATANA - water activation facility at JSI
TRIGA, Part I: Final design and activity calculations.” Nuclear Engineering and Technology (2024).](https://doi.org/10.1016/j.net.2024.09.036)

[2] [D. Kotnik, J. Peric, D. Govekar, L. Snoj, and I. Lengar. “KATANA - water activation facility at JSI
TRIGA, Part II: First experiments." Nuclear Engineering and Technology (2024).](https://doi.org/10.1016/j.net.2024.10.052)

[3] [J. Peric, D. Kotnik, L. Snoj, and V. Radulovi´c. “Neutron emission from water activation: Experiments and modeling under fusion-relevant conditions at the KATANA facility.” Fusion Engineering and Design, volume 216, p. 115052 (2025).](https://doi.org/10.1016/j.fusengdes.2025.115052)
  


  
