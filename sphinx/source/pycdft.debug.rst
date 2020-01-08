pycdft.debug
============

pycdft.debug.plot\_debug
------------------------

This module is for examining the:
 - Hirshfeld weight, :math:`w(\bf{r})`: ``get_hirsh``, ``get_hirsh_ct``
 - Hirshfeld weight gradient, :math:`\nabla w(\bf{r})`: ``get_grad``
 - charge densities for atom *i*, :math:`\rho_i(\bf{r})`: ``get_rho_atom``, ``get_rho``


Example usage is:

.. code-block:: python

   origin=tuple(np.multiply([-20.6689, -20.6689, -23.6216],0.529))
   get_hirsh_ct(solver1,origin)

where ``origin`` (in Bohr) can be extracted from any ``.cube`` file

.. automodule:: pycdft.debug.plot_debug
   :members:
   :undoc-members:
   :show-inheritance:

