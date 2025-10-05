# KATANA computational tool

Using Python to predict the activity inside the measurement snail in the KATANA water activation experiment. 
Reaction rates in irradiation volume calculated with MCNP.

The \textit{KATANA computational tool} models the water activation loop using a conventional analytical approach, divided into four sections: the irradiation region (inner irradiation Snail), the observation region (outer Snail No. 1), and two transport regions (pipes and pump). The circuit is discretised into uniform volume elements of 21.65 cm$^3$, which are sequentially transported through the simulated loop. For each time step $\Delta t$, the specific activity of a given isotope in each volume element is updated according to


$A'_{t+\Delta t} = A'_t\ e^{-\lambda \Delta t} + R(1-e^{-\lambda \Delta t})$,


\noindent where $A'_t$ is the isotope-specific activity at time $t$, $\lambda$ is the decay constant, and $R$ is the average reaction rate within the region of interest. The latter is defined as  

\begin{equation}
\label{eq_4_2}
R = \int \Phi(E)\,\sigma(E)\,N\,dE.
\end{equation}

\noindent Here, $\Phi(E)$ denotes the neutron flux, $\sigma(E)$ the microscopic cross section, and $N$ the atomic number density of the parent nuclide, calculated using the JSI TRIGA MCNP model \cite{Kotnik2024-KATANA_part_I}. The transport of each volume element proceeds in discrete steps determined by the volumetric flow rate.

![KATANA computational tool schematica](figures/25_10_02_katana_tool_scheme.png)

- Activity versus time (24_11_15_irradiation_scenario_pulse_model_V1_test.py)
  


  
