# Kernel

[Particula Index](../../../../README.md#particula-index) / [Particula](../../../index.md#particula) / [Next](../../index.md#next) / [Dynamics](../index.md#dynamics) / [Coagulation](./index.md#coagulation) / Kernel

> Auto-generated documentation for [particula.next.dynamics.coagulation.kernel](../../../../../particula/next/dynamics/coagulation/kernel.py) module.

- [Kernel](#kernel)
  - [CoulombDyachkov2007](#coulombdyachkov2007)
    - [CoulombDyachkov2007().dimensionless](#coulombdyachkov2007()dimensionless)
  - [CoulombGatti2008](#coulombgatti2008)
    - [CoulombGatti2008().dimensionless](#coulombgatti2008()dimensionless)
  - [CoulombGopalakrishnan2012](#coulombgopalakrishnan2012)
    - [CoulombGopalakrishnan2012().dimensionless](#coulombgopalakrishnan2012()dimensionless)
  - [CoulumbChahl2019](#coulumbchahl2019)
    - [CoulumbChahl2019().dimensionless](#coulumbchahl2019()dimensionless)
  - [HardSphere](#hardsphere)
    - [HardSphere().dimensionless](#hardsphere()dimensionless)
  - [KernelStrategy](#kernelstrategy)
    - [KernelStrategy().dimensionless](#kernelstrategy()dimensionless)
    - [KernelStrategy().kernel](#kernelstrategy()kernel)

## CoulombDyachkov2007

[Show source in kernel.py:129](../../../../../particula/next/dynamics/coagulation/kernel.py#L129)

Dyachkov et al. (2007) approximation for the dimensionless coagulation
kernel. Accounts for the Coulomb potential between particles.

#### References

-----------
- Dyachkov, S. A., Kustova, E. V., & Kustov, A. V. (2007). Coagulation of
particles in the transition regime: The effect of the Coulomb potential.
Journal of Chemical Physics, 126(12).
https://doi.org/10.1063/1.2713719

#### Signature

```python
class CoulombDyachkov2007(KernelStrategy): ...
```

#### See also

- [KernelStrategy](#kernelstrategy)

### CoulombDyachkov2007().dimensionless

[Show source in kernel.py:142](../../../../../particula/next/dynamics/coagulation/kernel.py#L142)

#### Signature

```python
def dimensionless(
    self,
    diffusive_knudsen: NDArray[np.float_],
    coulomb_potential_ratio: NDArray[np.float_],
) -> NDArray[np.float_]: ...
```



## CoulombGatti2008

[Show source in kernel.py:153](../../../../../particula/next/dynamics/coagulation/kernel.py#L153)

Gatti and Kortshagen (2008) approximation for the dimensionless coagulation
kernel. Accounts for the Coulomb potential between particles.

#### References

-----------
- Gatti, M., & Kortshagen, U. (2008). Analytical model of particle
charging in plasmas over a wide range of collisionality. Physical Review
E - Statistical, Nonlinear, and Soft Matter Physics, 78(4).
https://doi.org/10.1103/PhysRevE.78.046402

#### Signature

```python
class CoulombGatti2008(KernelStrategy): ...
```

#### See also

- [KernelStrategy](#kernelstrategy)

### CoulombGatti2008().dimensionless

[Show source in kernel.py:166](../../../../../particula/next/dynamics/coagulation/kernel.py#L166)

#### Signature

```python
def dimensionless(
    self,
    diffusive_knudsen: NDArray[np.float_],
    coulomb_potential_ratio: NDArray[np.float_],
) -> NDArray[np.float_]: ...
```



## CoulombGopalakrishnan2012

[Show source in kernel.py:177](../../../../../particula/next/dynamics/coagulation/kernel.py#L177)

Gopalakrishnan and Hogan (2012) approximation for the dimensionless
coagulation kernel. Accounts for the Coulomb potential between particles.

#### References

-----------
- Gopalakrishnan, R., & Hogan, C. J. (2012). Coulomb-influenced collisions
in aerosols and dusty plasmas. Physical Review E - Statistical, Nonlinear,
and Soft Matter Physics, 85(2).
https://doi.org/10.1103/PhysRevE.85.026410

#### Signature

```python
class CoulombGopalakrishnan2012(KernelStrategy): ...
```

#### See also

- [KernelStrategy](#kernelstrategy)

### CoulombGopalakrishnan2012().dimensionless

[Show source in kernel.py:190](../../../../../particula/next/dynamics/coagulation/kernel.py#L190)

#### Signature

```python
def dimensionless(
    self,
    diffusive_knudsen: NDArray[np.float_],
    coulomb_potential_ratio: NDArray[np.float_],
) -> NDArray[np.float_]: ...
```



## CoulumbChahl2019

[Show source in kernel.py:201](../../../../../particula/next/dynamics/coagulation/kernel.py#L201)

Chahl and Gopalakrishnan (2019) approximation for the dimensionless
coagulation kernel. Accounts for the Coulomb potential between particles.

#### References

-----------
- Chahl, H. S., & Gopalakrishnan, R. (2019). High potential, near free
molecular regime Coulombic collisions in aerosols and dusty plasmas.
Aerosol Science and Technology, 53(8), 933-957.
https://doi.org/10.1080/02786826.2019.1614522

#### Signature

```python
class CoulumbChahl2019(KernelStrategy): ...
```

#### See also

- [KernelStrategy](#kernelstrategy)

### CoulumbChahl2019().dimensionless

[Show source in kernel.py:214](../../../../../particula/next/dynamics/coagulation/kernel.py#L214)

#### Signature

```python
def dimensionless(
    self,
    diffusive_knudsen: NDArray[np.float_],
    coulomb_potential_ratio: NDArray[np.float_],
) -> NDArray[np.float_]: ...
```



## HardSphere

[Show source in kernel.py:116](../../../../../particula/next/dynamics/coagulation/kernel.py#L116)

Hard sphere dimensionless coagulation strategy.

#### Signature

```python
class HardSphere(KernelStrategy): ...
```

#### See also

- [KernelStrategy](#kernelstrategy)

### HardSphere().dimensionless

[Show source in kernel.py:121](../../../../../particula/next/dynamics/coagulation/kernel.py#L121)

#### Signature

```python
def dimensionless(
    self, diffusive_knudsen: NDArray[np.float_], coulomb_potential_ratio: ignore
) -> NDArray[np.float_]: ...
```



## KernelStrategy

[Show source in kernel.py:12](../../../../../particula/next/dynamics/coagulation/kernel.py#L12)

Abstract class for dimensionless coagulation strategies. This class defines
the dimensionless kernel (H) method that must be implemented by any
dimensionless coagulation strategy.

#### Methods

--------
- dimensionless (abstractmethod): Calculate the dimensionless coagulation
kernel.
- `-` *kernel* - Calculate the dimensioned coagulation kernel.

#### Signature

```python
class KernelStrategy(ABC): ...
```

### KernelStrategy().dimensionless

[Show source in kernel.py:25](../../../../../particula/next/dynamics/coagulation/kernel.py#L25)

Return the dimensionless coagulation kernel (H)

#### Arguments

-----
- `-` *diffusive_knudsen* - The diffusive Knudsen number (K_nD)
[dimensionless].
- `-` *coulomb_potential_ratio* - The Coulomb potential ratio (phi_E)
[dimensionless].

#### Returns

--------
The dimensionless coagulation kernel (H) [dimensionless].

#### References

-----------
- Dyachkov, S. A., Kustova, E. V., & Kustov, A. V. (2007). Coagulation
of particles in the transition regime: The effect of the Coulomb
potential. Journal of Chemical Physics, 126(12).
https://doi.org/10.1063/1.2713719
- Gatti, M., & Kortshagen, U. (2008). Analytical model of particle
charging in plasmas over a wide range of collisionality. Physical
Review E - Statistical, Nonlinear, and Soft Matter Physics, 78(4).
https://doi.org/10.1103/PhysRevE.78.046402
- Gopalakrishnan, R., & Hogan, C. J. (2011). Determination of the
transition regime collision kernel from mean first passage times.
Aerosol Science and Technology, 45(12), 1499-1509.
https://doi.org/10.1080/02786826.2011.601775
- Gopalakrishnan, R., & Hogan, C. J. (2012). Coulomb-influenced
collisions in aerosols and dusty plasmas. Physical Review E -
Statistical, Nonlinear, and Soft Matter Physics, 85(2).
https://doi.org/10.1103/PhysRevE.85.026410
- Chahl, H. S., & Gopalakrishnan, R. (2019). High potential, near free
molecular regime Coulombic collisions in aerosols and dusty plasmas.
Aerosol Science and Technology, 53(8), 933-957.
https://doi.org/10.1080/02786826.2019.1614522

#### Signature

```python
@abstractmethod
def dimensionless(
    self,
    diffusive_knudsen: NDArray[np.float_],
    coulomb_potential_ratio: NDArray[np.float_],
) -> NDArray[np.float_]: ...
```

### KernelStrategy().kernel

[Show source in kernel.py:68](../../../../../particula/next/dynamics/coagulation/kernel.py#L68)

The dimensioned coagulation kernel for each particle pair, calculated
from the dimensionless coagulation kernel and the reduced quantities.
All inputs are square matrices, for all particle-particle interactions.

#### Arguments

-----
- `-` *dimensionless_kernel* - The dimensionless coagulation kernel
[dimensionless].
- `-` *coulomb_potential_ratio* - The Coulomb potential ratio [dimensionless].
- `-` *sum_of_radii* - The sum of the radii of the particles [m].
- `-` *reduced_mass* - The reduced mass of the particles [kg].
- `-` *reduced_friction_factor* - The reduced friction factor of the
particles [dimensionless].

#### Returns

--------
The dimensioned coagulation kernel, as a square matrix, of all
particle-particle interactions [m^3/s].

Check, were the /s comes from.

#### References

-----------

#### Signature

```python
def kernel(
    self,
    dimensionless_kernel: NDArray[np.float_],
    coulomb_potential_ratio: NDArray[np.float_],
    sum_of_radii: NDArray[np.float_],
    reduced_mass: NDArray[np.float_],
    reduced_friction_factor: NDArray[np.float_],
) -> NDArray[np.float_]: ...
```