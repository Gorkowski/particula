# Cloud Droplet Coagulation Kernel 

Here, we discuss the implementation of the geometric collision kernel for cloud droplets as described in Part II by Ayala et al. (2008). Part I provides a detailed explanation of the direct numerical simulations. Where as Part II is the parameterization of the collision kernel for cloud droplets in turbulent flows. The implementation involves calculating the geometric collision rate of sedimenting droplets based on the turbulent flow properties and droplet characteristics.

Ayala, O., Rosa, B., Wang, L. P., & Grabowski, W. W. (2008). Effects of turbulence on the geometric collision rate of sedimenting droplets. Part 1. Results from direct numerical simulation. New Journal of Physics, 10. https://doi.org/10.1088/1367-2630/10/7/075015

Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on the geometric collision rate of sedimenting droplets. Part 2. Theory and parameterization. New Journal of Physics, 10. https://doi.org/10.1088/1367-2630/10/7/075016

## **Derived Geometric Collision Kernel Implementation**

This implementation is based on the parameterized equations described in the document and focuses on calculating the geometric collision kernel $\beta_{12}$ for cloud droplets.

## Outline

- [Cloud Droplet Coagulation Kernel](#cloud-droplet-coagulation-kernel)
  - [**Derived Geometric Collision Kernel Implementation**](#derived-geometric-collision-kernel-implementation)
  - [Outline](#outline)
    - [**Variable Descriptions**](#variable-descriptions)
    - [**1. Droplet Properties**](#1-droplet-properties)
    - [**2. Turbulent Flow Properties**](#2-turbulent-flow-properties)
      - [$g$: Gravitational Acceleration](#g-gravitational-acceleration)
    - [Derived Parameters](#derived-parameters)
      - [$\\tau\_k$: Kolmogorov Time](#tau_k-kolmogorov-time)
      - [$\\eta$: Kolmogorov Length Scale](#eta-kolmogorov-length-scale)
      - [$v\_k$: Kolmogorov Velocity Scale](#v_k-kolmogorov-velocity-scale)
      - [$u'$: Fluid RMS Fluctuation Velocity](#u-fluid-rms-fluctuation-velocity)
      - [$T\_L$: Lagrangian Integral Scale](#t_l-lagrangian-integral-scale)
      - [$L\_e$: Eulerian Integral Scale](#l_e-eulerian-integral-scale)
      - [$a\_o$: Coefficient](#a_o-coefficient)
      - [$\\tau\_T$: Lagrangian Taylor Microscale Time](#tau_t-lagrangian-taylor-microscale-time)
      - [$\\lambda$: Taylor Microscale](#lambda-taylor-microscale)
      - [**Droplet Inertial Response Time**](#droplet-inertial-response-time)
      - [**Particle Terminal Velocity**](#particle-terminal-velocity)
      - [**Particle Reynolds Number**](#particle-reynolds-number)
      - [**Stokes Number**](#stokes-number)
  - [**Step 3: Radial Relative Velocity ($\\langle |w\_r| \\rangle$)**](#step-3-radial-relative-velocity-langle-w_r-rangle)
  - [**Step 4: Radial Distribution Function ($g\_{12}$)**](#step-4-radial-distribution-function-g_12)
    - [**Step 5: Collision Kernel**](#step-5-collision-kernel)


### **Variable Descriptions**

Here are the variables, their definitions, and concepts with additional explanations and links for further reading:

---

### **1. Droplet Properties**

- $a_1, a_2$: Radii of the droplets. These determine size-dependent properties such as droplet inertia and terminal velocity. 

- $\rho_w$: Density of water. The mass per unit volume of water, typically $1000 \, \text{kg/m}^3$. It is essential for calculating droplet inertia and terminal velocity.

- $\rho$: Density of air. The mass per unit volume of air, affecting drag and settling velocity. Typical sea-level values are around $1.225 \, \text{kg/m}^3$.

- $\nu$: Kinematic viscosity. The ratio of dynamic viscosity to fluid density, quantifying resistance to flow.

- $\tau_p$: Droplet inertial response time. The characteristic time it takes for a droplet to adjust to changes in the surrounding airflow, critical for droplet motion analysis.

- $v'^{(i)}_p$: Particle RMS fluctuation velocity. The root mean square of the fluctuating velocity component, representing variability in turbulent flow.

- $f_u$: Particle response coefficient. Measures how particles respond to fluid velocity fluctuations, helping quantify their turbulent motion.

- $f(R)$: Spatial correlation coefficient. Describes the correlation of fluid velocities at two points separated by a distance $R$, influencing droplet interactions.

- $g_{12}$: Radial distribution function (RDF). A measure of how particle pairs are spatially distributed due to turbulence and gravity.

---

### **2. Turbulent Flow Properties**

- $u$: Local air velocity. The instantaneous velocity of air at a given point. Turbulence causes $u$ to vary in space and time.

- $\varepsilon$: Turbulence dissipation rate. The rate at which turbulent kinetic energy is converted into thermal energy per unit mass.

- $R_\lambda$: Reynolds number. A dimensionless number that characterizes the flow regime, depending on turbulence intensity and scale.

- $\lambda_D$: Longitudinal Taylor-type microscale. A characteristic length scale of fluid acceleration in turbulence, related to energy dissipation and viscosity.

- $T_L$: Lagrangian integral scale. The timescale over which fluid particles maintain velocity correlations, describing large-scale turbulence behavior.

- $u'$: Fluid RMS fluctuation velocity. The root mean square of fluid velocity fluctuations, characterizing turbulence intensity.

- $S$: Skewness of longitudinal velocity gradient. A measure of asymmetry in velocity gradient fluctuations, significant for small-scale turbulence analysis.

- $Y^f(t)$: Fluid Lagrangian trajectory. The path traced by a fluid particle as it moves through turbulence.

- $\tau_T$: Lagrangian Taylor microscale time. A timescale describing the decay of velocity correlation along a fluid particle trajectory.

#### $g$: Gravitational Acceleration

The acceleration due to gravity, approximately 9.81 \, $\text{m/s}^2$ on Earth's surface. This force drives droplet sedimentation in turbulent air.

---

### Derived Parameters

#### $\tau_k$: Kolmogorov Time

The smallest timescale in turbulence where viscous forces dominate:
$$
\tau_k = \left(\frac{\nu}{\varepsilon}\right)^{1/2}
$$

#### $\eta$: Kolmogorov Length Scale

The smallest scale in turbulence:
$$
\eta = \left(\frac{\nu^3}{\varepsilon}\right)^{1/4}
$$

#### $v_k$: Kolmogorov Velocity Scale
A velocity scale related to the smallest turbulent eddies:
$$
v_k = (\nu \varepsilon)^{1/4}
$$

#### $u'$: Fluid RMS Fluctuation Velocity

Quantifies turbulence intensity:
$$
u' = \frac{R_\lambda^{1/2} v_k}{15^{1/4}} 
$$

#### $T_L$: Lagrangian Integral Scale

Describes large-scale turbulence:
$$
T_L = \frac{u'^2}{\epsilon}
$$

#### $L_e$: Eulerian Integral Scale

Length scale for large eddies:
$$
L_e = 0.5 \frac{u'^3}{\epsilon}
$$

#### $a_o$: Coefficient

A Reynolds-dependent parameter:
$$
a_o = \frac{11+7 R_\lambda}{205 + R_\lambda}
$$

#### $\tau_T$: Lagrangian Taylor Microscale Time

Time correlation decay for turbulent trajectories:
$$
\tau_T = \tau_k \left(\frac{2 R_\lambda}{15^{1/2} a_o}\right)^{1/2}
$$

#### $\lambda$: Taylor Microscale

Length scale linked to fluid flow:
$$
\lambda = u' \left(\frac{15 \nu^2}{\epsilon}\right)^{1/2}
$$

#### **Droplet Inertial Response Time**

Adjusts droplet inertia:
$$
\tau_p = \frac{2}{9} \frac{\rho_w}{\rho} \frac{a^2}{\nu f(Re_p)}
$$
with:
$$
f(Re_p) = 1 + 0.15 Re_p^{0.687}
$$

#### **Particle Terminal Velocity**

The settling velocity under gravity:
$$
v_p = \tau_p |g|
$$

#### **Particle Reynolds Number**

Characterizes droplet flow:
$$
Re_p = \frac{2 a v_p}{\nu}
$$

#### **Stokes Number**

Non-dimensional inertia parameter:
$$
St = \frac{\tau_p}{\tau_k}
$$

---

## **Step 3: Radial Relative Velocity ($\langle |w_r| \rangle$)**
Use:
$$
\langle |w_r| \rangle = \sqrt{\frac{2}{\pi}} \sigma f(b)
$$
where:
- $f(b) = \frac{1}{2}\sqrt{\pi}\left(b + \frac{0.5}{b}\right)\text{erf}(b) + \frac{1}{2}\exp(-b^2)$
- $b = \frac{g|\tau_{p1} - \tau_{p2}|}{\sqrt{2} \sigma}$
- $\sigma$: variance of relative velocity (derived via equations for turbulent velocity components).

## **Step 4: Radial Distribution Function ($g_{12}$)**
Use an empirical formula for the RDF:
$$
g_{12}(R) = \left(\frac{\eta^2 + r_c^2}{R^2 + r_c^2}\right)^{C_1/2}
$$
where $C_1$ and $r_c$ are derived based on droplet and turbulence properties.

### **Step 5: Collision Kernel**
Combine $\langle |w_r| \rangle$ and $g_{12}$ into:
$$
\beta_{12} = 2\pi R^2 \langle |w_r| \rangle g_{12}
$$
where $R = a_1 + a_2$ is the sum of droplet radii.

