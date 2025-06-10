from typing import Union

import numpy as np
from casadi import *
from do_mpc.data import load_results
from do_mpc.model import Model


class POD:
    def __init__(
        self,
        nx_unique: int,
    ):
        """A class that applies `Proper Orthogonal Decomposition` (POD) to reduce your original model, that may be expensive to evaluate.
        To bake your rigorous model, snapshot data of its origial states and inputs is needed.
        The reduction is done by calling the method "reduce()" passing the original model. It returns the reduced order model (ROM).
        Your original approximated states are stored and accessible with their original names as auxillary expressions
        for monitoring or to incorporate them into constraints.

        To simulate, the initial state must be projected to the reduced space using the method "map()" before it
        is passed to the simulator or mpc-object.
        """
        self.nx_unique = nx_unique  # The number of unique states
        self.scales = None  # The scales used to scale each unique state
        self.r = None  # The rank of the reduced system
        self.phi = None  # The trucated orthogonal projection matrix phi
        self.U = None  # The full orthonormal matrix U of the SVD decomposition that contains the spatial modes phi(z)
        self.S = None  # The singular values of the data.
        self.V_T = None  # The orthonormal matrix V_T of the decomposition that contains the temporal coefficients a(t)

    def perform_svd(self, snaps: np.array):
        """Perform Singular Value Decomposition on the data."""
        scaled_snaps = self.set_scaling(x=snaps)
        self.U, self.S, self.V_T = np.linalg.svd(scaled_snaps)

    def load_state_space_snaps(self, snaps_path: str) -> np.array:
        data = load_results(file_name=snaps_path)
        if "simulator" in data.keys():
            print(f"Loading state space snaps from {snaps_path}.")
            state_space_snapshots = data["simulator"]["_x"].T
            return state_space_snapshots
        else:
            raise ValueError(f"No simulator data found in the data file {snaps_path}.")

    def set_scaling(self, x: np.array) -> np.array:
        """Scales a vector of all states in the original space to values between -1 and 1 using a max scaling.
        Creates predefined scale() and invert_scale() methods that memorize and replicate/invert the scaling procedure.
        """
        nx_total = x.shape[0]
        assert (
            nx_total % self.nx_unique == 0
        ), f"The number of total states in the input ({x.shape[0]}) cannot correspond to the number of unique states ({self.nx_unique})."
        block_size = nx_total // self.nx_unique
        x_scaled = np.zeros_like(x)
        scales = np.zeros(shape=(nx_total, 1))
        for i in range(self.nx_unique):
            block = x[i * block_size : (i + 1) * block_size, :]
            absmax = np.max(np.abs(block))
            x_scaled[i * block_size : (i + 1) * block_size, :] = block / absmax
            scales[i * block_size : (i + 1) * block_size, :] = 1 / absmax
        self.scales = scales
        return x_scaled

    def truncate(self, r: int = None, e: float = None) -> np.array:
        """Truncates the spatial mode matrix U to yield the reduced projection matrix phi.
        Either provide the reduced rank 'r' directly or give an information threshold 'e'.
        """
        if e is not None:
            rel_energy = 0
            e_tot = np.sum(self.S**2)
            for r, sigma_j in enumerate(self.S):
                rel_energy += sigma_j**2 / e_tot
                if rel_energy >= e:
                    break
        self.r = r
        phi = self.U[:, :r]
        self.phi = phi
        return phi

    def map(self, x: Union[np.array, SX, MX]) -> Union[np.array, SX, MX]:
        """
        Maps a state vector from full space to the reduced space.
        """
        assert x.shape == (
            self.phi.shape[0],
            1,
        ), f"Full state vector x must be of shape {(self.phi.shape[0], 1)} to match the data. It is of shape {x.shape}."
        x_scaled = x * self.scales
        return self.phi.T @ x_scaled

    def unmap(self, x_tilde: Union[np.array, SX, MX]) -> Union[np.array, SX, MX]:
        """
        Maps a state vector back to the original full space from the reduced space.
        """
        assert x_tilde.shape == (
            self.phi.shape[1],
            1,
        ), f"The inner product of the reduced state vector ({x_tilde.shape}) cannot be done using the projection matrix ({self.phi.shape})."
        x_scaled = self.phi @ x_tilde
        x = x_scaled / self.scales
        return x

    def reduce(self, rigorous_model: Model, e: float = None, r: int = None) -> Model:
        """Reduces the state space via Proper Orthogonal Decomposition from rank n to lower rank r."""
        assert (
            rigorous_model.flags["setup"] is True
        ), "The original model has not been set up. Pass a model that has been set up."
        if r is None:
            if e is None:
                raise ValueError(
                    "Provide either the information threshold 'e' or the reduced rank 'r' directly to determine the projection matrix 'phi'."
                )
        self.truncate(e=e, r=r)
        nx_model, nx_data = rigorous_model.n_x, self.phi.shape[0]
        assert (
            nx_model == nx_data
        ), f"The number of states in the model ({nx_model}) does not match the number of states arising from the data matrix ({nx_data})."

        print("Baking reduced order model.")
        rom = Model(model_type=rigorous_model.model_type)
        x_tld = rom.set_variable(var_type="_x", var_name="x_tld", shape=(r, 1))
        x_approx = self.unmap(x_tilde=x_tld)
        # the approximated full states, shape = (n_x, 1)

        # Transfer the original model's variables and expressions to the ROM
        for key in rigorous_model.u.keys()[1:]:
            rom.set_variable("_u", key, shape=rigorous_model.u[key].shape)
        for key in rigorous_model.tvp.keys()[1:]:
            rom.set_variable("_tvp", key, shape=rigorous_model.tvp[key].shape)
        for key in rigorous_model.p.keys()[1:]:
            rom.set_variable("_p", key, shape=rigorous_model.p[key].shape)
        # TODO: Double check copying of the noise. This may be not correct.
        # TODO: Copy measurements
        for key in rigorous_model.w.keys()[1:]:
            rom.set_variable("_w", key, shape=rigorous_model.p[key].shape)
        for key in rigorous_model.aux.keys()[1:]:
            expr_fun = rigorous_model._aux_expression_fun
            expr = expr_fun(x_approx, rom.u, rom.z, rom.tvp, rom.p)
            expr_struct = rigorous_model._aux_expression(expr)
            rom.set_expression(key, expr_struct[key])

        block_size = nx_model // self.nx_unique
        state_keys = rigorous_model.x.keys()
        for j, key in enumerate(state_keys):
            # set them as auxillary expressions to be able to set contraints in the original space
            # Note, they are also set as vectorized expressions according to the original state vectors.
            rom.set_expression(
                expr_name=key, expr=x_approx[j * block_size : (j + 1) * block_size]
            )
        rhs_fun = rigorous_model._rhs_fun
        rhs_expr = rhs_fun(x_approx, rom.u, rom.z, rom.tvp, rom.p, rom.w)
        # map the full rhs back to the reduced space
        rhs_expr = self.map(x=rhs_expr)
        rom.set_rhs(var_name="x_tld", expr=rhs_expr)
        return rom

    # ---------------------------------------
    # Methods for analysis
    # ---------------------------------------

    def plot_singular_values(self, log: bool = True):
        """Creates a minimal plot of the singular values of the data."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.scatter(
            x=range(len(self.S)),
            y=self.S,
        )
        ax.set_xlabel(r"$r$")
        ax.set_ylabel(r"$\sigma_i$ / -")
        ax.annotate(
            f"$r = ${self.r}",
            xy=(0.7, 0.8),
            xycoords="axes fraction",
        )
        if log:
            ax.set_yscale("log")
        return fig

    def plot_modes(self):
        """Plots the spatial and temporal modes of the POD"""
        import matplotlib.cm as cm
        import matplotlib.pyplot as plt

        cmap = cm.get_cmap("viridis")

        fig, axes = plt.subplots(2, 1, figsize=(10, 7))
        for i in range(self.r):
            color = cmap(i / (self.r - 1))
            axes[0].plot(self.U[:, i], c=color, label=f"$z$-mode {i}")
            axes[1].plot(self.V_T[i, :], c=color, label=f"$t$-mode {i}")

        axes[0].legend()
        axes[0].set_ylabel("$\\phi(z)$")
        axes[0].set_xlabel("$x_i$")
        axes[1].legend()
        axes[1].set_ylabel("$a(t)$")
        axes[1].set_xlabel("$t_i$")
        return fig
