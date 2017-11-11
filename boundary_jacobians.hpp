/* Copyright (C) 2017 Jerry Watkins
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/* Slip-wall and Symmetry Boundary Condition */
template <size_t nVars, size_t nDims>
#ifdef _GPU
__device__ __forceinline__
#endif
void compute_Slip_Jac(double U[nVars], double Ub[nVars], double dUbdU[nVars][nVars], 
	double norm[nDims])
{
	if (nDims == 2)
	{
		double nx = norm[0];
		double ny = norm[1];

		/* Primitive Variables */
		double uL = U[1] / U[0];
		double vL = U[2] / U[0];

		double uR = Ub[1] / Ub[0];
		double vR = Ub[2] / Ub[0];

		/* Compute dUbdU */
		dUbdU[0][0] = 1;
		dUbdU[3][0] = 0.5 * (uL*uL + vL*vL - uR*uR - vR*vR);

		dUbdU[1][1] = 1.0-nx*nx;
		dUbdU[2][1] = -nx*ny;
		dUbdU[3][1] = -uL + (1.0-nx*nx)*uR - nx*ny*vR;

		dUbdU[1][2] = -nx*ny;
		dUbdU[2][2] = 1.0-ny*ny;
		dUbdU[3][2] = -vL - nx*ny*uR + (1.0-ny*ny)*vR;

		dUbdU[3][3] = 1;
	}

	else if (nDims == 3)
	{
		double nx = norm[0];
		double ny = norm[1];
		double nz = norm[2];

		/* Primitive Variables */
		double uL = U[1] / U[0];
		double vL = U[2] / U[0];
		double wL = U[3] / U[0];

		double uR = Ub[1] / Ub[0];
		double vR = Ub[2] / Ub[0];
		double wR = Ub[3] / Ub[0];

		/* Compute dUbdU */
		dUbdU[0][0] = 1;
		dUbdU[4][0] = 0.5 * (uL*uL + vL*vL + wL*wL - uR*uR - vR*vR - wR*wR);

		dUbdU[1][1] = 1.0-nx*nx;
		dUbdU[2][1] = -nx*ny;
		dUbdU[3][1] = -nx*nz;
		dUbdU[4][1] = -uL + (1.0-nx*nx)*uR - nx*ny*vR - nx*nz*wR;

		dUbdU[1][2] = -nx*ny;
		dUbdU[2][2] = 1.0-ny*ny;
		dUbdU[3][2] = -ny*nz;
		dUbdU[4][2] = -vL - nx*ny*uR + (1.0-ny*ny)*vR - ny*nz*wR;

		dUbdU[1][3] = -nx*nz;
		dUbdU[2][3] = -ny*nz;
		dUbdU[3][3] = 1.0-nz*nz;
		dUbdU[4][3] = -wL - nx*nz*uR - ny*nz*vR + (1.0-nz*nz)*wR;

		dUbdU[4][4] = 1;
	}
}

/* No-Slip Adiabatic Wall */
template <size_t nVars, size_t nDims>
#ifdef _GPU
__device__ __forceinline__
#endif
void compute_NoSlip_Jac(double U[nVars], double Q[nVars][nDims], double dUbdU[nVars][nVars], 
	double dQbdU[nDims][nVars][nVars], double dQbdQ[nDims][nDims][nVars][nVars], 
  double norm[nDims])
{
  /* Matrix Parameters */
  double astar[nDims][nVars-1] = {0};
  for (unsigned int dimi = 0; dimi < nDims; dimi++)
    for (unsigned int dimj = 0; dimj < nDims; dimj++)
      for (unsigned int var = 0; var < nVars-1; var++)
        astar[dimi][var] += norm[dimi] * norm[dimj] * Q[var][dimj] / U[0];

  double momF = 0.0;
  for (unsigned int dim = 0; dim < nDims; dim++)
    momF += U[dim+1] * U[dim+1];
  momF /= U[0];
  double estar1 = (U[nDims+1] - 2.0 * momF) / U[0];
  double estar2 = (U[nDims+1] - momF) / U[0];

	if (nDims == 2)
	{
		double nx = norm[0];
		double ny = norm[1];

		/* Primitive Variables */
		double uL = U[1] / U[0];
		double vL = U[2] / U[0];

		/* Compute dUbdU */
		dUbdU[0][0] = 1;
		dUbdU[3][0] = 0.5 * (uL*uL + vL*vL);

		dUbdU[3][1] = -uL;

		dUbdU[3][2] = -vL;

		dUbdU[3][3] = 1;

    /* Compute dQbdU */
    for (unsigned int dim = 0; dim < nDims; dim++)
    {
      dQbdU[dim][3][0] = estar1 * astar[dim][0] + uL * astar[dim][1] + vL * astar[dim][2];
      dQbdU[dim][3][1] = astar[dim][1] - 2.0 * uL * astar[dim][0];
      dQbdU[dim][3][2] = astar[dim][2] - 2.0 * vL * astar[dim][0];
      dQbdU[dim][3][3] = astar[dim][0];
    }

		/* Compute dQxR/dQxL */
		dQbdQ[0][0][0][0] = 1;
		dQbdQ[0][0][3][0] = nx*nx * estar2;

		dQbdQ[0][0][1][1] = 1;
		dQbdQ[0][0][3][1] = nx*nx * uL;

		dQbdQ[0][0][2][2] = 1;
		dQbdQ[0][0][3][2] = nx*nx * vL;

		dQbdQ[0][0][3][3] = 1.0 - nx*nx;

		/* Compute dQyR/dQxL */
		dQbdQ[1][0][3][0] = nx*ny * estar2;
		dQbdQ[1][0][3][1] = nx*ny * uL;
		dQbdQ[1][0][3][2] = nx*ny * vL;
		dQbdQ[1][0][3][3] = -nx * ny;

		/* Compute dQxR/dQyL */
		dQbdQ[0][1][3][0] = nx*ny * estar2;
		dQbdQ[0][1][3][1] = nx*ny * uL;
		dQbdQ[0][1][3][2] = nx*ny * vL;
		dQbdQ[0][1][3][3] = -nx * ny;

		/* Compute dQyR/dQyL */
		dQbdQ[1][1][0][0] = 1;
		dQbdQ[1][1][3][0] = ny*ny * estar2;

		dQbdQ[1][1][1][1] = 1;
		dQbdQ[1][1][3][1] = ny*ny * uL;

		dQbdQ[1][1][2][2] = 1;
		dQbdQ[1][1][3][2] = ny*ny * vL;

		dQbdQ[1][1][3][3] = 1.0 - ny*ny;
  }

  else if (nDims == 3)
	{
		double nx = norm[0];
		double ny = norm[1];
		double nz = norm[2];

		/* Primitive Variables */
		double uL = U[1] / U[0];
		double vL = U[2] / U[0];
		double wL = U[3] / U[0];

		/* Compute dUbdU */
		dUbdU[0][0] = 1;
		dUbdU[4][0] = 0.5 * (uL*uL + vL*vL + wL*wL);

		dUbdU[4][1] = -uL;

		dUbdU[4][2] = -vL;

		dUbdU[4][3] = -wL;

		dUbdU[4][4] = 1;

    /* Compute dQbdU */
    for (unsigned int dim = 0; dim < nDims; dim++)
    {
      dQbdU[dim][4][0] = estar1 * astar[dim][0] + uL * astar[dim][1] + 
        vL * astar[dim][2] + wL * astar[dim][3];
      dQbdU[dim][4][1] = astar[dim][1] - 2.0 * uL * astar[dim][0];
      dQbdU[dim][4][2] = astar[dim][2] - 2.0 * vL * astar[dim][0];
      dQbdU[dim][4][3] = astar[dim][3] - 2.0 * wL * astar[dim][0];
      dQbdU[dim][4][4] = astar[dim][0];
    }

		/* Compute dQxR/dQxL */
		dQbdQ[0][0][0][0] = 1;
		dQbdQ[0][0][4][0] = nx*nx * estar2;

		dQbdQ[0][0][1][1] = 1;
		dQbdQ[0][0][4][1] = nx*nx * uL;

		dQbdQ[0][0][2][2] = 1;
		dQbdQ[0][0][4][2] = nx*nx * vL;

		dQbdQ[0][0][3][3] = 1;
		dQbdQ[0][0][4][3] = nx*nx * wL;

		dQbdQ[0][0][4][4] = 1.0 - nx*nx;

		/* Compute dQyR/dQxL */
		dQbdQ[1][0][4][0] = nx*ny * estar2;
		dQbdQ[1][0][4][1] = nx*ny * uL;
		dQbdQ[1][0][4][2] = nx*ny * vL;
		dQbdQ[1][0][4][3] = nx*ny * wL;
		dQbdQ[1][0][4][4] = -nx * ny;

		/* Compute dQzR/dQxL */
		dQbdQ[2][0][4][0] = nx*nz * estar2;
		dQbdQ[2][0][4][1] = nx*nz * uL;
		dQbdQ[2][0][4][2] = nx*nz * vL;
		dQbdQ[2][0][4][3] = nx*nz * wL;
		dQbdQ[2][0][4][4] = -nx * nz;

		/* Compute dQxR/dQyL */
		dQbdQ[0][1][4][0] = nx*ny * estar2;
		dQbdQ[0][1][4][1] = nx*ny * uL;
		dQbdQ[0][1][4][2] = nx*ny * vL;
		dQbdQ[0][1][4][3] = nx*ny * wL;
		dQbdQ[0][1][4][4] = -nx * ny;

		/* Compute dQyR/dQyL */
		dQbdQ[1][1][0][0] = 1;
		dQbdQ[1][1][4][0] = ny*ny * estar2;

		dQbdQ[1][1][1][1] = 1;
		dQbdQ[1][1][4][1] = ny*ny * uL;

		dQbdQ[1][1][2][2] = 1;
		dQbdQ[1][1][4][2] = ny*ny * vL;

		dQbdQ[1][1][3][3] = 1;
		dQbdQ[1][1][4][3] = ny*ny * wL;

		dQbdQ[1][1][4][4] = 1.0 - ny*ny;

		/* Compute dQzR/dQyL */
		dQbdQ[2][1][4][0] = nz*ny * estar2;
		dQbdQ[2][1][4][1] = nz*ny * uL;
		dQbdQ[2][1][4][2] = nz*ny * vL;
		dQbdQ[2][1][4][3] = nz*ny * wL;
		dQbdQ[2][1][4][4] = -nz * ny;

		/* Compute dQxR/dQzL */
		dQbdQ[0][2][4][0] = nx*nz * estar2;
		dQbdQ[0][2][4][1] = nx*nz * uL;
		dQbdQ[0][2][4][2] = nx*nz * vL;
		dQbdQ[0][2][4][3] = nx*nz * wL;
		dQbdQ[0][2][4][4] = -nx * nz;

		/* Compute dQyR/dQzL */
		dQbdQ[1][2][4][0] = ny*nz * estar2;
		dQbdQ[1][2][4][1] = ny*nz * uL;
		dQbdQ[1][2][4][2] = ny*nz * vL;
		dQbdQ[1][2][4][3] = ny*nz * wL;
		dQbdQ[1][2][4][4] = -ny * nz;

		/* Compute dQzR/dQzL */
		dQbdQ[2][2][0][0] = 1;
		dQbdQ[2][2][4][0] = nz*nz * estar2;

		dQbdQ[2][2][1][1] = 1;
		dQbdQ[2][2][4][1] = nz*nz * uL;

		dQbdQ[2][2][2][2] = 1;
		dQbdQ[2][2][4][2] = nz*nz * vL;

		dQbdQ[2][2][3][3] = 1;
		dQbdQ[2][2][4][3] = nz*nz * wL;

		dQbdQ[2][2][4][4] = 1.0 - nz*nz;
  }
}

/* Characteristic Riemann Invariant Far Field */
template <size_t nVars, size_t nDims>
#ifdef _GPU
__device__ __forceinline__
#endif
void compute_Char_Jac(double U[nVars], double Ub[nVars], double dUbdU[nVars][nVars], 
	double dQbdQ[nDims][nDims][nVars][nVars], double norm[nDims], double V_fs[nDims]
	double rho_fs, double P_fs, double gamma, bool viscous)
{
	/* Compute wall normal velocities */
	double VnL = 0.0; double VnR = 0.0;
	for (unsigned int dim = 0; dim < nDims; dim++)
	{
		VnL += U[dim+1] / U[0] * norm[dim];
		VnR += V_fs[dim] * norm[dim];
	}

	/* Compute pressure */
	double momF = 0.0;
	for (unsigned int dim = 0; dim < nDims; dim++)
	{
		momF += U[dim + 1] * U[dim + 1];
	}

	momF /= U[0];

	double PL = (gamma - 1.0) * (U[nDims + 1] - 0.5 * momF);
	double PR = P_fs;

	double cL = std::sqrt(gamma * PL / U[0]);
	double cR = std::sqrt(gamma * PR / rho_fs);

	/* Compute Riemann Invariants */
	double RL = VnL + 2.0 / (gamma - 1) * cL;
	double RB = VnR - 2.0 / (gamma - 1) * cR;

	double cstar = 0.25 * (gamma - 1) * (RL - RB);
	double ustarn = 0.5 * (RL + RB);

	if (nDims == 2)
	{
		double nx = norm[0];
		double ny = norm[1];
		double gam = gamma;

		/* Primitive Variables */
		double rhoL = U[0];
		double uL = U[1] / U[0];
		double vL = U[2] / U[0];

		double rhoR = Ub[0];
		double uR = Ub[1] / Ub[0];
		double vR = Ub[2] / Ub[0];

		if (VnL < 0.0) /* Case 1: Inflow */
		{
			/* Matrix Parameters */
			double a1 = 0.5 * rhoR / cstar;
			double a2 = gam / (rhoL * cL);
			
			double b1 = -VnL / rhoL - a2 / rhoL * (PL / (gam-1.0) - 0.5 * momF);
			double b2 = nx / rhoL - a2 * uL;
			double b3 = ny / rhoL - a2 * vL;
			double b4 = a2 / cstar;

			double c1 = cstar * cstar / ((gam-1.0) * gam) + 0.5 * (uR*uR + vR*vR);
			double c2 = uR * nx + vR * ny + cstar / gam;

			/* Compute dUbdU */
			dUbdU[0][0] = a1 * b1;
			dUbdU[1][0] = a1 * b1 * uR + 0.5 * rhoR * b1 * nx;
			dUbdU[2][0] = a1 * b1 * vR + 0.5 * rhoR * b1 * ny;
			dUbdU[3][0] = a1 * b1 * c1 + 0.5 * rhoR * b1 * c2;

			dUbdU[0][1] = a1 * b2;
			dUbdU[1][1] = a1 * b2 * uR + 0.5 * rhoR * b2 * nx;
			dUbdU[2][1] = a1 * b2 * vR + 0.5 * rhoR * b2 * ny;
			dUbdU[3][1] = a1 * b2 * c1 + 0.5 * rhoR * b2 * c2;

			dUbdU[0][2] = a1 * b3;
			dUbdU[1][2] = a1 * b3 * uR + 0.5 * rhoR * b3 * nx;
			dUbdU[2][2] = a1 * b3 * vR + 0.5 * rhoR * b3 * ny;
			dUbdU[3][2] = a1 * b3 * c1 + 0.5 * rhoR * b3 * c2;

			dUbdU[0][3] = 0.5 * rhoR * b4;
			dUbdU[1][3] = 0.5 * rhoR * (b4 * uR + a2 * nx);
			dUbdU[2][3] = 0.5 * rhoR * (b4 * vR + a2 * ny);
			dUbdU[3][3] = 0.5 * rhoR * (b4 * c1 + a2 * c2);
		}

		else  /* Case 2: Outflow */
		{
			/* Matrix Parameters */
			double a1 = gam * rhoR / (gam-1.0);
			double a2 = gam / (rhoL * cL);
			double a3 = (gam-1.0) / (gam * PL);
			double a4 = (gam-1.0) / (2.0 * gam * cstar);
			double a5 = rhoR * cstar * cstar / (gam-1.0) / (gam-1.0);
			double a6 = rhoR * cstar / (2.0 * gam);

			double b1 = -VnL / rhoL - a2 / rhoL * (PL / (gam-1.0) - 0.5 * momF);
			double b2 = nx / rhoL - a2 * uL;
			double b3 = ny / rhoL - a2 * vL;

			double c1 = 0.5 * b1 * nx - (-VnL * nx + uL) / rhoL;
			double c2 = 0.5 * b2 * nx + (1.0 - nx*nx) / rhoL;
			double c3 = 0.5 * b3 * nx - nx * ny / rhoL;
			double c4 = ustarn * nx + uL - VnL * nx;

			double d1 = 0.5 * b1 * ny - (-VnL * ny + vL) / rhoL;
			double d2 = 0.5 * b2 * ny - nx * ny / rhoL;
			double d3 = 0.5 * b3 * ny + (1.0 - ny*ny) / rhoL;
			double d4 = ustarn * ny + vL - VnL * ny;

			double e1 = 1.0 / rhoL - 0.5 * a3 * momF / rhoL + a4 * b1;
			double e2 = a3 * uL + a4 * b2;
			double e3 = a3 * vL + a4 * b3;
			double e4 = -a3 + a2 * a4;

			double f1 = 0.5 * a1 * (c4*c4 + d4*d4) + a5;

			/* Compute dUbdU */
			dUbdU[0][0] = a1 * e1;
			dUbdU[1][0] = a1 * e1 * c4 + rhoR * c1;
			dUbdU[2][0] = a1 * e1 * d4 + rhoR * d1;
			dUbdU[3][0] = rhoR * (c1*c4 + d1*d4) + e1 * f1 + a6 * b1;

			dUbdU[0][1] = a1 * e2;
			dUbdU[1][1] = a1 * e2 * c4 + rhoR * c2;
			dUbdU[2][1] = a1 * e2 * d4 + rhoR * d2;
			dUbdU[3][1] = rhoR * (c2*c4 + d2*d4) + e2 * f1 + a6 * b2;

			dUbdU[0][2] = a1 * e3;
			dUbdU[1][2] = a1 * e3 * c4 + rhoR * c3;
			dUbdU[2][2] = a1 * e3 * d4 + rhoR * d3;
			dUbdU[3][2] = rhoR * (c3*c4 + d3*d4) + e3 * f1 + a6 * b3;

			dUbdU[0][3] = a1 * e4;
			dUbdU[1][3] = a1 * e4 * c4 + 0.5 * rhoR * a2 * nx;
			dUbdU[2][3] = a1 * e4 * d4 + 0.5 * rhoR * a2 * ny;
			dUbdU[3][3] = 0.5 * rhoR * a2 * (c4*nx + d4*ny) + e4 * f1 + a2 * a6;
		}
	}

	else if (nDims == 3)
	{
		double nx = norm[0];
		double ny = norm[1];
		double nz = norm[2];
		double gam = gamma;

		/* Primitive Variables */
		double rhoL = U[0];
		double uL = U[1] / U[0];
		double vL = U[2] / U[0];
		double wL = U[3] / U[0];

		double rhoR = Ub[0];
		double uR = Ub[1] / Ub[0];
		double vR = Ub[2] / Ub[0];
		double wR = Ub[3] / Ub[0];

		if (VnL < 0.0) /* Case 1: Inflow */
		{
			/* Matrix Parameters */
			double a1 = 0.5 * rhoR / cstar;
			double a2 = gam / (rhoL * cL);
			
			double b1 = -VnL / rhoL - a2 / rhoL * (PL / (gam-1.0) - 0.5 * momF);
			double b2 = nx / rhoL - a2 * uL;
			double b3 = ny / rhoL - a2 * vL;
			double b4 = nz / rhoL - a2 * wL;
			double b5 = a2 / cstar;

			double c1 = cstar * cstar / ((gam-1.0) * gam) + 0.5 * (uR*uR + vR*vR + wR*wR);
			double c2 = uR * nx + vR * ny + wR * nz + cstar / gam;

			/* Compute dUbdU */
			dUbdU[0][0] = a1 * b1;
			dUbdU[1][0] = a1 * b1 * uR + 0.5 * rhoR * b1 * nx;
			dUbdU[2][0] = a1 * b1 * vR + 0.5 * rhoR * b1 * ny;
			dUbdU[3][0] = a1 * b1 * wR + 0.5 * rhoR * b1 * nz;
			dUbdU[4][0] = a1 * b1 * c1 + 0.5 * rhoR * b1 * c2;

			dUbdU[0][1] = a1 * b2;
			dUbdU[1][1] = a1 * b2 * uR + 0.5 * rhoR * b2 * nx;
			dUbdU[2][1] = a1 * b2 * vR + 0.5 * rhoR * b2 * ny;
			dUbdU[3][1] = a1 * b2 * wR + 0.5 * rhoR * b2 * nz;
			dUbdU[4][1] = a1 * b2 * c1 + 0.5 * rhoR * b2 * c2;

			dUbdU[0][2] = a1 * b3;
			dUbdU[1][2] = a1 * b3 * uR + 0.5 * rhoR * b3 * nx;
			dUbdU[2][2] = a1 * b3 * vR + 0.5 * rhoR * b3 * ny;
			dUbdU[3][2] = a1 * b3 * wR + 0.5 * rhoR * b3 * nz;
			dUbdU[4][2] = a1 * b3 * c1 + 0.5 * rhoR * b3 * c2;

			dUbdU[0][3] = a1 * b4;
			dUbdU[1][3] = a1 * b4 * uR + 0.5 * rhoR * b4 * nx;
			dUbdU[2][3] = a1 * b4 * vR + 0.5 * rhoR * b4 * ny;
			dUbdU[3][3] = a1 * b4 * wR + 0.5 * rhoR * b4 * nz;
			dUbdU[4][3] = a1 * b4 * c1 + 0.5 * rhoR * b4 * c2;

			dUbdU[0][4] = 0.5 * rhoR * b5;
			dUbdU[1][4] = 0.5 * rhoR * (b5 * uR + a2 * nx);
			dUbdU[2][4] = 0.5 * rhoR * (b5 * vR + a2 * ny);
			dUbdU[3][4] = 0.5 * rhoR * (b5 * wR + a2 * nz);
			dUbdU[4][4] = 0.5 * rhoR * (b5 * c1 + a2 * c2);
		}

		else  /* Case 2: Outflow */
		{
			/* Matrix Parameters */
			double a1 = gam * rhoR / (gam-1.0);
			double a2 = gam / (rhoL * cL);
			double a3 = (gam-1.0) / (gam * PL);
			double a4 = (gam-1.0) / (2.0 * gam * cstar);
			double a5 = rhoR * cstar * cstar / (gam-1.0) / (gam-1.0);
			double a6 = rhoR * cstar / (2.0 * gam);

			double b1 = -VnL / rhoL - a2 / rhoL * (PL / (gam-1.0) - 0.5 * momF);
			double b2 = nx / rhoL - a2 * uL;
			double b3 = ny / rhoL - a2 * vL;
			double b4 = nz / rhoL - a2 * wL;

			double c1 = 0.5 * b1 * nx - (-VnL * nx + uL) / rhoL;
			double c2 = 0.5 * b2 * nx + (1.0 - nx*nx) / rhoL;
			double c3 = 0.5 * b3 * nx - nx * ny / rhoL;
			double c4 = 0.5 * b4 * nx - nx * nz / rhoL;
			double c5 = ustarn * nx + uL - VnL * nx;

			double d1 = 0.5 * b1 * ny - (-VnL * ny + vL) / rhoL;
			double d2 = 0.5 * b2 * ny - nx * ny / rhoL;
			double d3 = 0.5 * b3 * ny + (1.0 - ny*ny) / rhoL;
			double d4 = 0.5 * b4 * ny - ny * nz / rhoL;
			double d5 = ustarn * ny + vL - VnL * ny;

			double e1 = 0.5 * b1 * nz - (-VnL * nz + wL) / rhoL;
			double e2 = 0.5 * b2 * nz - nx * nz / rhoL;
			double e3 = 0.5 * b3 * nz - ny * nz / rhoL;
			double e4 = 0.5 * b4 * nz + (1.0 - nz*nz) / rhoL;
			double e5 = ustarn * nz + wL - VnL * nz;

			double f1 = 1.0 / rhoL - 0.5 * a3 * momF / rhoL + a4 * b1;
			double f2 = a3 * uL + a4 * b2;
			double f3 = a3 * vL + a4 * b3;
			double f4 = a3 * wL + a4 * b4;
			double f5 = -a3 + a2 * a4;

			double g1 = 0.5 * a1 * (c5*c5 + d5*d5 + e5*e5) + a5;

			/* Compute dUbdU */
			dUbdU[0][0] = a1 * f1;
			dUbdU[1][0] = a1 * f1 * c5 + rhoR * c1;
			dUbdU[2][0] = a1 * f1 * d5 + rhoR * d1;
			dUbdU[3][0] = a1 * f1 * e5 + rhoR * e1;
			dUbdU[4][0] = rhoR * (c1*c5 + d1*d5 + e1*e5) + f1 * g1 + a6 * b1;

			dUbdU[0][1] = a1 * f2;
			dUbdU[1][1] = a1 * f2 * c5 + rhoR * c2;
			dUbdU[2][1] = a1 * f2 * d5 + rhoR * d2;
			dUbdU[3][1] = a1 * f2 * e5 + rhoR * e2;
			dUbdU[4][1] = rhoR * (c2*c5 + d2*d5 + e2*e5) + f2 * g1 + a6 * b2;

			dUbdU[0][2] = a1 * f3;
			dUbdU[1][2] = a1 * f3 * c5 + rhoR * c3;
			dUbdU[2][2] = a1 * f3 * d5 + rhoR * d3;
			dUbdU[3][2] = a1 * f3 * e5 + rhoR * e3;
			dUbdU[4][2] = rhoR * (c3*c5 + d3*d5 + e3*e5) + f3 * g1 + a6 * b3;

			dUbdU[0][3] = a1 * f4;
			dUbdU[1][3] = a1 * f4 * c5 + rhoR * c4;
			dUbdU[2][3] = a1 * f4 * d5 + rhoR * d4;
			dUbdU[3][3] = a1 * f4 * e5 + rhoR * e4;
			dUbdU[4][3] = rhoR * (c4*c5 + d4*d5 + e4*e5) + f4 * g1 + a6 * b4;

			dUbdU[0][4] = a1 * f5;
			dUbdU[1][4] = a1 * f5 * c5 + 0.5 * rhoR * a2 * nx;
			dUbdU[2][4] = a1 * f5 * d5 + 0.5 * rhoR * a2 * ny;
			dUbdU[3][4] = a1 * f5 * e5 + 0.5 * rhoR * a2 * nz;
			dUbdU[4][4] = 0.5 * rhoR * a2 * (c5*nx + d5*ny + e5*nz) + f5 * g1 + a2 * a6;
		}
	}

	/* Extrapolate gradients */
	if (viscous)
		for (unsigned int dim = 0; dim < nDims; dim++)
			for (unsigned int var = 0; var < nVars; var++)
				dQbdQ[dim][dim][var][var] = 1;
}

