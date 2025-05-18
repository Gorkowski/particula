
import taichi as ti


@ti.data_oriented
class FieldIO:
    """
    Helper for copying NumPy arrays into Taichi fields (1D/2D only).

    Supports assignment of 1D and 2D float64 arrays to Taichi fields.
    Used for efficient data transfer between NumPy and Taichi.

    Examples:
        ```py
        field_io = _FieldIO()
        field_io.from_numpy(ti_field, np_array)
        ```
    """

    @ti.kernel
    def _assign_1d(
        self,
        field: ti.template(),
        array: ti.types.ndarray(dtype=ti.f64, ndim=1),
    ):
        """
        Assign a 1D NumPy array to a Taichi field.

        Args:
            - field : Taichi field to assign to.
            - array : 1D NumPy array of float64.

        Returns:
            - None
        """
        for i in field:
            field[i] = array[i]

    @ti.kernel
    def _assign_2d(
        self,
        field: ti.template(),
        array: ti.types.ndarray(dtype=ti.f64, ndim=2),
    ):
        """
        Assign a 2D NumPy array to a Taichi field.

        Args:
            - field : Taichi field to assign to.
            - array : 2D NumPy array of float64.

        Returns:
            - None
        """
        for i, j in field:
            field[i, j] = array[i, j]

    def from_numpy(self, field, array):
        """
        Copy a NumPy array (1D or 2D) into a Taichi field.

        Args:
            - field : Taichi field to assign to.
            - array : NumPy array (1D or 2D) of float64.

        Returns:
            - None

        Raises:
            - ValueError : If array is not 1D or 2D.
        """
        if array.ndim == 1:
            self._assign_1d(field, array)
        elif array.ndim == 2:
            self._assign_2d(field, array)
        else:
            raise ValueError("Only 1-D/2-D supported")
