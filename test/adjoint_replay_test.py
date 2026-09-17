# Copyright (C) 2026 by David Lilien <dlilien@iu.edu>
#
# This file is part of icepack.
#
# icepack is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# The full text of the license can be found in the file LICENSE in the
# icepack source directory or at <http://www.gnu.org/licenses/>.

import firedrake
from firedrake import inner, as_vector, dx
import firedrake.adjoint
from firedrake.adjoint import Control, ReducedFunctional
import icepack


def test_replay_with_changing_inflow():
    r"""Replaying the tape has to reproduce the forward model when the inflow
    velocity is changed between diagnostic solves

    The flow solver keeps one Dirichlet BC on a field that it updates in
    place, while firedrake replays a solve with the BC object of the live
    solver rather than its taped value."""
    Lx, Ly = 20e3, 20e3
    mesh = firedrake.RectangleMesh(8, 8, Lx, Ly)
    Q = firedrake.FunctionSpace(mesh, "CG", 2)
    V = firedrake.VectorFunctionSpace(mesh, "CG", 2)
    x, y = firedrake.SpatialCoordinate(mesh)

    h = firedrake.Function(Q).interpolate(500 - 100 * x / Lx)
    inflows = [
        firedrake.Function(V).interpolate(as_vector((u_in + 50 * x / Lx, 0)))
        for u_in in (100.0, 150.0)
    ]

    model = icepack.models.IceShelf()
    opts = {
        "dirichlet_ids": [1],
        "side_wall_ids": [3, 4],
        "diagnostic_solver_type": "petsc",
    }
    solver = icepack.solvers.FlowSolver(model, **opts)

    firedrake.adjoint.continue_annotation()
    tape = firedrake.adjoint.get_working_tape()
    tape.clear_tape()
    try:
        A = firedrake.Function(Q).assign(icepack.rate_factor(255.0))
        J = 0.0
        u = inflows[0].copy(deepcopy=True)
        for u_inflow in inflows:
            u = solver.diagnostic_solve(
                velocity=u, thickness=h, fluidity=A, velocity_inflow=u_inflow
            )
            J += firedrake.assemble(inner(u, u) * dx)
        rf = ReducedFunctional(J, Control(A))
    finally:
        firedrake.adjoint.pause_annotation()

    try:
        assert abs(rf(A) - J) < 1e-8 * abs(J)

        dA = firedrake.Function(Q).interpolate(0.1 * A * (1 + x / Lx))
        assert firedrake.adjoint.taylor_test(rf, A, dA) > 1.9
    finally:
        tape.clear_tape()
