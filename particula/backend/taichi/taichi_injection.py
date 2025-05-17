"""
Minimal demonstration of “dependency injection” with Taichi data-oriented
classes.  The Particle and GasSpecies objects are created in Python, injected
into PressureDeltaCalc, and their @ti.func methods are called from inside a
kernel.
"""

import taichi as ti
import numpy as np

ti.init(arch=ti.cpu)  # use ti.cpu if you don’t have a GPU


# --------------------------------------------------------------------------- #
#  Supporting objects that will be “injected”
# --------------------------------------------------------------------------- #
@ti.data_oriented
class GasSpecies:
    def __init__(self, molar_mass, henry_const):
        self.molar_mass = ti.field(ti.f64, shape=())
        self.henry_const = ti.field(ti.f64, shape=())
        self.molar_mass[None] = molar_mass
        self.henry_const[None] = henry_const

    @ti.func
    def pure_vapor_pressure(self, T):
        # toy Antoine-like equation
        return ti.exp(10.0 - self.molar_mass[None] / T)

    @ti.func
    def partial_pressure(self, T):
        # Henry’s-law placeholder
        return self.henry_const[None] * T


@ti.data_oriented
class Particle:
    def __init__(self, masses_np):
        n = masses_np.size
        self.mass = ti.field(dtype=ti.f64, shape=(n,))
        self.mass.from_numpy(masses_np)

    @ti.func
    def species_mass(self, i):
        return self.mass[i]

    @ti.func
    def kelvin_term(self, r, molar_mass, conc, T):
        R = 8.314  # J mol⁻¹ K⁻¹
        sigma = 0.072  # N m⁻¹  (surface tension, placeholder)
        return ti.exp(2 * sigma * molar_mass / (r * R * T))

    @ti.func
    def partial_pressure(self, p0, conc):
        gamma = 1.0  # activity coefficient (placeholder)
        return gamma * conc * p0


# --------------------------------------------------------------------------- #
#  Calculator that “owns” the collaborators inside its kernel
# --------------------------------------------------------------------------- #
@ti.data_oriented
class PressureDeltaCalc:
    def __init__(self, particle: Particle, gas: GasSpecies, radii_np):
        self.p = particle  # injected dependency
        self.g = gas  # injected dependency
        self.n = radii_np.size
        self.r = ti.field(dtype=ti.f64, shape=(self.n,))
        self.delta = ti.field(dtype=ti.f64, shape=(self.n,))
        self.r.from_numpy(radii_np)

    @ti.kernel
    def compute(self, T: ti.f64):
        for i in range(self.n):
            mcp = self.p.species_mass(i)
            pvp = self.g.pure_vapor_pressure(T)
            p_par = self.p.partial_pressure(pvp, mcp)
            p_gas = self.g.partial_pressure(T)
            kelv = self.p.kelvin_term(
                self.r[i], self.g.molar_mass[None], mcp, T
            )
            self.delta[i] = p_gas - p_par * kelv


# --------------------------------------------------------------------------- #
#  Simple test driver
# --------------------------------------------------------------------------- #
def main():
    # toy data for a 4-particle system
    n = 4
    masses = np.array([1e-18, 2e-18, 3e-18, 4e-18])  # kg (placeholder)
    radii = np.linspace(50e-9, 200e-9, n)  # m
    T = 298.15  # K

    particle = Particle(masses)
    gas = GasSpecies(molar_mass=0.018, henry_const=1.5e-3)
    calc = PressureDeltaCalc(particle, gas, radii)

    calc.compute(T)
    print("Delta P (gas - particle * Kelvin) [Pa]:")
    print(calc.delta.to_numpy())


if __name__ == "__main__":
    main()
