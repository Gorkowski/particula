# Code-Naming Guide for Particula & Related Projects

Update the code following the rules below.
Use these rules to keep names predictable, readable, and lint-friendly (≤ 79 chars, snake\_case).

---

## 1.  General Conventions

| Kind                     | Style                                 | Notes / Examples                                  |
| ------------------------ | ------------------------------------- | ------------------------------------------------- |
| **Modules & files**      | `snake_case.py`                       | `mass_transfer.py`, `ti_knudsen_number_module.py` |
| **Functions & methods**  | `snake_case()`                        | `get_mass_transfer_rate()`                        |
| **Variables**            | `snake_case`                          | `particle_radius`, `vapor_transition`             |
| **Classes**              | `PascalCase`                          | `TiKnudsenNumber`                                 |
| **Constants**            | `UPPER_SNAKE_CASE`                    | `GAS_CONSTANT = 8.314`                            |
| **Type aliases**         | `PascalCase`                          | `NdArrayFloat = NDArray[np.float64]`              |
| **Private (internal)**   | prefix with `_`                       | `_scaling_factor`                                 |
| **Taichi kernels/funcs** | `kget_*`, `fget_*` + snake\_case args | `kget_mass_transfer_rate(radius, …)`              |

---

## 2.  Specific Guidelines

1. **Be explicit, avoid abbreviations**
   *Bad:* `r`, `M`, `dmt` → *Good:* `particle_radius`, `molar_mass`, `mass_transfer_rate`.

2. **Units in the name only when ambiguity exists**
   *Good:* `time_step_s` if seconds matter.

3. **Boolean flags start with verbs**
   `update_gases`, `use_turbulence`.

4. **Arrays vs scalars**
   *Optional but helpful*: suffix `_array` or `_vec` if both forms appear (`molar_mass_array` vs `molar_mass`).

5. **Dimensions first, then property**
   `n_particles`, `n_species`, `n_bins`.

6. **Error / uncertainty**
   `error_molar_mass`, `std_mixing_state`.

7. **Taichi helper naming**

   * Function returning a Taichi func: `fget_knudsen_number`
   * Function returning a Taichi kernel: `kget_knudsen_number`
   * Class with Tiaichi @ti.data_oriented: `TiKnudsenNumber`

---

## 3.  Tiny Rename Examples

```diff
- r = 1e-6          # radius [m]
+ particle_radius = 1e-6

- M = 0.02897       # kg/mol
+ molar_mass = 0.02897

- temp = 300
+ temperature = 300

- dmt = dm_dt(...)
+ mass_transfer_rate = get_mass_transfer_rate(...)
```

---

## 4.  Quick-Reference Name Glossary

* particle_radius
* molar_mass
* density
* diffusion_coefficient
* vapor_transition
* pressure_delta
* first_order_mass_transport
* mass_rate
* radius_transfer_rate
* particle_concentration
* gas_mass
* particle_mass
* time_step
* knudsen_number
* accommodation_coefficient
* mean_free_path
* hygroscopicity_kappa
* light_scattering_cross_section

Stick to these roots and build longer names by combining them (e.g., `error_particle_radius`, `n_particle_bins`).

# Docstring Specification: Classes, Attributes, Methods, and Functions

**Instruction**  
Analyze all  definitions provided and improve their docstrings for clarity, consistency, and adherence to best practices.

---

## High-Level Objectives

1. **Improve Documentation Clarity**  
   - Ensure all function docstrings follow a clear, structured format.  
   - Use concise parameter descriptions and clearly stated return values.

2. **Include Mathematical Equations**  
   - When applicable, present equations in **Unicode format** for broader compatibility.

3. **Ensure Consistency**  
   - Maintain uniform style for docstrings: proper indentation, spacing, formatting.

4. **Add Examples**  
   - Provide code snippets to demonstrate usage of classes and methods.

5. **Include References**  
   - Insert a **"References"** section where needed, citing reliable sources (e.g., Wikipedia, journal articles, or books).

---

## Mid-Level Objectives

1. **Standardized Docstring Format**  
   Use the following template as a guide:

   ```python
   class ExampleClassName:
       """
       Short description of what the class is or does.

       Longer description of the class, including its purpose and functionality.
       This can be multiple lines. Discuss why the class is important and how it
       fits into a larger API or system.

       Attributes:
           - param1 : Description of param1.
           - param2 : Description of param2.

       Methods:
        - method_name: Brief description or context.
        - another_method: Brief description or context.

       Examples:
           ```py title="Example Usage"
           import particula as par
           example_object = par.ExampleClassName(param1, param2)
           output = example_object.method_name(value1, value2)
           # Output: ...
           ```

       References:
           - Author Name, "Title of the Article," Journal Name,
             Volume, Issue, Year.
             [DOI](link)
           - "Article Title,"
             [Wikipedia](URL).
       """

       def __init__(self, param1, param2):
           """
           Initialize the ExampleClassName with parameters.

           Arguments:
               - param1 : Description of param1.
               - param2 : Description of param2.

           Returns:
               - None
           """
           self.param1 = param1
           self.param2 = param2

       def method_name(self, value1, value2):
           """
           Brief description of what the method does.

           A longer description of the method, including its purpose
           and methodology. Can be multiple lines. For example:

           - φ = (γ × β) / c
               - φ is Description of φ.
               - γ is Description of γ.
               - β is Description of β.
               - c is Description of the constant.

           Arguments:
               - value1 : Description of value1.
               - value2 : Description of value2.

           Returns:
               - Description of the return value.

           Examples:
               ```py title="Example"
               example_object.method_name(2, 3)
               # Output: 1.5
               ```

               ```py title="Example Usage with Arrays"
               example_object.method_name(np.array([4,5,5]), np.array([2,3,4]))
               # Output: array([4.0, 1.66666667, 1.25])
               ```

           References:
               - Author Name, "Title of the Article," Journal Name,
                 Volume, Issue, Year.
                 [DOI](link)
               - "Article Title,"
                 [Wikipedia](link).
            """
            return (value1 * value2) / self.param1
           ```

2. **Mathematical Equation Representation**  
   - Include relevant mathematical equations in **Unicode format** (e.g., `C = (P × M) / (R × T)`).

3. **Consistent Spacing and Formatting**  
   - Insert a **space** after each colon in parameter descriptions (`- parameter : Description`).  
   - Maintain proper line breaks and indentation.  
   - Provide usage examples under an **"Examples"** subheading.  
   - Include a **"References"** section whenever citing sources.

---

## Implementation Steps

1. **Analyze Existing Docstrings**  
   - Identify missing or incorrect parameter names.  
   - Check for inconsistent formatting, unclear descriptions, or missing references.

2. **Update Docstrings for Clarity**  
   - Use a single, consistent style for parameter listings: `- parameter : Description`.  
   - Verify each parameter and return value is accurately described.  
   - Keep line lengths under 79 characters when possible.  
   - Use equations in Unicode format where relevant.  
   - Add references for scientific or technical validity.

3. **Apply Consistency Rules**  
   - All arguments must follow the `parameter : Description` style.  
   - Equations should be in Unicode format.  
   - Insert a "References" section if any sources are used.  
   - Ensure docstrings are consistently structured.
