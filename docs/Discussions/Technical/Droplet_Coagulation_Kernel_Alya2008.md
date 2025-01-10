# Cloud Droplet Coagulation Kernel 

Here, we discuss the implementation of the geometric collision kernel for cloud droplets as described in Part II by Ayala et al. (2008). Part II provides a detailed explanation of the theory and parameterization of the collision kernel for cloud droplets in turbulent flows. The implementation involves calculating the geometric collision rate of sedimenting droplets based on the turbulent flow properties and droplet characteristics.

Ayala, O., Rosa, B., Wang, L. P., & Grabowski, W. W. (2008). Effects of turbulence on the geometric collision rate of sedimenting droplets. Part 1. Results from direct numerical simulation. New Journal of Physics, 10. https://doi.org/10.1088/1367-2630/10/7/075015

Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on the geometric collision rate of sedimenting droplets. Part 2. Theory and parameterization. New Journal of Physics, 10. https://doi.org/10.1088/1367-2630/10/7/075016

## **Derived Geometric Collision Kernel Implementation**

This implementation is based on the parameterized equations described in the document and focuses on calculating the geometric collision kernel $\beta_{12}$ for cloud droplets.

### **Step 1: Input Parameters**
1. Droplet properties:
   - Radii $a_1$ and $a_2$
   - Density of water, $\rho_w$
   - Air properties: density $\rho$, kinematic viscosity $\nu$
2. Turbulent flow properties:
   - Turbulence dissipation rate, $\varepsilon$
   - Reynolds number, $R_\lambda$
3. Gravitational acceleration, $g$

### **Step 2: Derived Parameters**
Using the input values, compute the following:
1. Kolmogorov length scale: 
   $$
   \eta = \left(\frac{\nu^3}{\varepsilon}\right)^{1/4}
   $$
2. Kolmogorov velocity scale:
   $$
   v_k = (\nu \varepsilon)^{1/4}
   $$
3. Particle response time:
   $$
   \tau_p = \frac{2}{9} \frac{\rho_w}{\rho} \frac{a^2}{\nu f(Re_p)}
   $$
    where $f(Re_p)$ is a function of the particle Reynolds number $Re_p$:
    $$
    f(Re_p) = 1 + 0.15 Re_p^{0.687}
    $$
4. Terminal velocity:
   $$
   v_p = \tau_p g
   $$
5. Stokes number:
   $$
   St = \frac{\tau_p}{\tau_k}
   $$

### **Step 3: Radial Relative Velocity ($\langle |w_r| \rangle$)**
Use:
$$
\langle |w_r| \rangle = \sqrt{\frac{2}{\pi}} \sigma f(b)
$$
where:
- $f(b) = \frac{1}{2}\sqrt{\pi}\left(b + \frac{0.5}{b}\right)\text{erf}(b) + \frac{1}{2}\exp(-b^2)$
- $b = \frac{g|\tau_{p1} - \tau_{p2}|}{\sqrt{2} \sigma}$
- $\sigma$: variance of relative velocity (derived via equations for turbulent velocity components).

### **Step 4: Radial Distribution Function ($g_{12}$)**
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

