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

/* Linear Advective Flux Jacobians */
template <size_t nVars, size_t nDims>
#ifdef _GPU
__device__ __forceinline__
#endif
void compute_dFdUadv_AdvDiff(double dFdU[nVars][nVars][nDims], double A[nDims])
{
  for (unsigned int dim = 0; dim < nDims; dim++)
    dFdU[0][0][dim] = A[dim];
}

/* Inviscid Flux Jacobians */
template <size_t nVars, size_t nDims>
#ifdef _GPU
__device__ __forceinline__
#endif
void compute_dFdUinv_EulerNS(double U[nVars], double dFdU[nVars][nVars][nDims], double gamma)
{
  if (nDims == 2)
  {
    /* Primitive Variables */
    double invrho = 1.0 / U[0];
    double u = U[1] * invrho;
    double v = U[2] * invrho;
    double e = U[3];

    /* Set inviscid dFdU values in the x-direction */
    dFdU[1][0][0] = 0.5 * ((gamma-3.0) * u*u + (gamma-1.0) * v*v);
    dFdU[2][0][0] = -u * v;
    dFdU[3][0][0] = -gamma * e * u * invrho + (gamma-1.0) * u * (u*u + v*v);

    dFdU[0][1][0] = 1;
    dFdU[1][1][0] = (3.0-gamma) * u;
    dFdU[2][1][0] = v;
    dFdU[3][1][0] = gamma * e * invrho + 0.5 * (1.0-gamma) * (3.0*u*u + v*v);

    dFdU[1][2][0] = (1.0-gamma) * v;
    dFdU[2][2][0] = u;
    dFdU[3][2][0] = (1.0-gamma) * u * v;

    dFdU[1][3][0] = (gamma-1.0);
    dFdU[3][3][0] = gamma * u;

    /* Set inviscid dFdU values in the y-direction */
    dFdU[1][0][1] = -u * v;
    dFdU[2][0][1] = 0.5 * ((gamma-1.0) * u*u + (gamma-3.0) * v*v);
    dFdU[3][0][1] = -gamma * e * v * invrho + (gamma-1.0) * v * (u*u + v*v);

    dFdU[1][1][1] = v;
    dFdU[2][1][1] = (1.0-gamma) * u;
    dFdU[3][1][1] = (1.0-gamma) * u * v;

    dFdU[0][2][1] = 1;
    dFdU[1][2][1] = u;
    dFdU[2][2][1] = (3.0-gamma) * v;
    dFdU[3][2][1] = gamma * e * invrho + 0.5 * (1.0-gamma) * (u*u + 3.0*v*v);

    dFdU[2][3][1] = (gamma-1.0);
    dFdU[3][3][1] = gamma * v;
  }

  else if (nDims == 3)
  {
    /* Primitive Variables */
    double invrho = 1.0 / U[0];
    double u = U[1] * invrho;
    double v = U[2] * invrho;
    double w = U[3] * invrho;
    double e = U[4];

    /* Set inviscid dFdU values in the x-direction */
    dFdU[1][0][0] = 0.5 * ((gamma-3.0) * u*u + (gamma-1.0) * (v*v + w*w));
    dFdU[2][0][0] = -u * v;
    dFdU[3][0][0] = -u * w;
    dFdU[4][0][0] = -gamma * e * u * invrho + (gamma-1.0) * u * (u*u + v*v + w*w);

    dFdU[0][1][0] = 1;
    dFdU[1][1][0] = (3.0-gamma) * u;
    dFdU[2][1][0] = v;
    dFdU[3][1][0] = w;
    dFdU[4][1][0] = gamma * e * invrho + 0.5 * (1.0-gamma) * (3.0*u*u + v*v + w*w);

    dFdU[1][2][0] = (1.0-gamma) * v;
    dFdU[2][2][0] = u;
    dFdU[4][2][0] = (1.0-gamma) * u * v;

    dFdU[1][3][0] = (1.0-gamma) * w;
    dFdU[3][3][0] = u;
    dFdU[4][3][0] = (1.0-gamma) * u * w;

    dFdU[1][4][0] = (gamma-1.0);
    dFdU[4][4][0] = gamma * u;

    /* Set inviscid dFdU values in the y-direction */
    dFdU[1][0][1] = -u * v;
    dFdU[2][0][1] = 0.5 * ((gamma-1.0) * (u*u + w*w) + (gamma-3.0) * v*v);
    dFdU[3][0][1] = -v * w;
    dFdU[4][0][1] = -gamma * e * v * invrho + (gamma-1.0) * v * (u*u + v*v + w*w);

    dFdU[1][1][1] = v;
    dFdU[2][1][1] = (1.0-gamma) * u;
    dFdU[4][1][1] = (1.0-gamma) * u * v;

    dFdU[0][2][1] = 1;
    dFdU[1][2][1] = u;
    dFdU[2][2][1] = (3.0-gamma) * v;
    dFdU[3][2][1] = w;
    dFdU[4][2][1] = gamma * e * invrho + 0.5 * (1.0-gamma) * (u*u + 3.0*v*v + w*w);

    dFdU[2][3][1] = (1.0-gamma) * w;
    dFdU[3][3][1] = v;
    dFdU[4][3][1] = (1.0-gamma) * v * w;

    dFdU[2][4][1] = (gamma-1.0);
    dFdU[4][4][1] = gamma * v;

    /* Set inviscid dFdU values in the z-direction */
    dFdU[1][0][2] = -u * w;
    dFdU[2][0][2] = -v * w;
    dFdU[3][0][2] = 0.5 * ((gamma-1.0) * (u*u + v*v) + (gamma-3.0) * w*w);
    dFdU[4][0][2] = -gamma * e * w * invrho + (gamma-1.0) * w * (u*u + v*v + w*w);

    dFdU[1][1][2] = w;
    dFdU[3][1][2] = (1.0-gamma) * u;
    dFdU[4][1][2] = (1.0-gamma) * u * w;

    dFdU[2][2][2] = w;
    dFdU[3][2][2] = (1.0-gamma) * v;
    dFdU[4][2][2] = (1.0-gamma) * v * w;

    dFdU[0][3][2] = 1;
    dFdU[1][3][2] = u;
    dFdU[2][3][2] = v;
    dFdU[3][3][2] = (3.0-gamma) * w;
    dFdU[4][3][2] = gamma * e * invrho + 0.5 * (1.0-gamma) * (u*u + v*v + 3.0*w*w);

    dFdU[3][4][2] = (gamma-1.0);
    dFdU[4][4][2] = gamma * w;
  }
}

/* Linear Diffusive Flux Jacobians */
template <size_t nVars, size_t nDims>
#ifdef _GPU
__device__ __forceinline__
#endif
void compute_dFdQdiff_AdvDiff(double dFdQ[nVars][nVars][nDims][nDims], double D)
{
  for (unsigned int dim = 0; dim < nDims; dim++)
    dFdQ[0][0][dim][dim] = -D;
}

/* Viscous Flux Jacobians */
template <size_t nVars, size_t nDims>
#ifdef _GPU
__device__ __forceinline__
#endif
void compute_dFdUvisc_EulerNS_add(double U[nVars], double Q[nVars][nDims], double dFdU[nVars][nVars][nDims], 
    double gamma, double prandtl, double mu)
{
  double invrho = 1.0 / U[0];
  double diffCo1 = mu * invrho;
  double diffCo2 = gamma * mu * invrho / prandtl;

  if (nDims == 2)
  {
    /* Primitive Variables */
    double u = U[1] * invrho;
    double v = U[2] * invrho;
    double e = U[3];

    /* Gradients */
    double rho_dx = Q[0][0];
    double momx_dx = Q[1][0];
    double momy_dx = Q[2][0];
    double e_dx = Q[3][0];
    
    double rho_dy = Q[0][1];
    double momx_dy = Q[1][1];
    double momy_dy = Q[2][1];
    double e_dy = Q[3][1];

    /* Set viscous dFdU values in the x-direction */
    dFdU[1][0][0] -= (2.0/3.0) * (4.0*u*rho_dx - 2.0*(v*rho_dy + momx_dx) + momy_dy) * invrho * diffCo1;
    dFdU[2][0][0] -= (2.0*(v*rho_dx + u*rho_dy) - (momx_dy + momy_dx)) * invrho * diffCo1;
    dFdU[3][0][0] -= (1.0/3.0) * (3.0*(4.0*u*u + 3.0*v*v)*rho_dx + 3.0*u*v*rho_dy - 4.0*u*(2.0*momx_dx - momy_dy) - 6.0*v*(momx_dy + momy_dx)) * invrho * diffCo1 + 
                                 (-e_dx + (2.0*e*invrho - 3.0*(u*u + v*v))*rho_dx + 2.0*(u*momx_dx + v*momy_dx)) * invrho * diffCo2;

    dFdU[1][1][0] -= -(4.0/3.0) * rho_dx * invrho * diffCo1;
    dFdU[2][1][0] -= -rho_dy * invrho * diffCo1;
    dFdU[3][1][0] -= -(1.0/3.0) * (8.0*u*rho_dx + v*rho_dy - 4.0*momx_dx + 2.0*momy_dy) * invrho * diffCo1 + 
                                  (2.0*u*rho_dx - momx_dx) * invrho * diffCo2;

    dFdU[1][2][0] -= (2.0/3.0) * rho_dy * invrho * diffCo1;
    dFdU[2][2][0] -= -rho_dx * invrho * diffCo1;
    dFdU[3][2][0] -= -(1.0/3.0) * (6.0*v*rho_dx + u*rho_dy - 3.0*(momx_dy + momy_dx)) * invrho * diffCo1 + 
                                  (2.0*v*rho_dx - momy_dx) * invrho * diffCo2;

    dFdU[3][3][0] -= -rho_dx * invrho * diffCo2;

    /* Set viscous dFdU values in the y-direction */
    dFdU[1][0][1] -= (2.0*(v*rho_dx + u*rho_dy) - (momx_dy + momy_dx)) * invrho * diffCo1;
    dFdU[2][0][1] -= (2.0/3.0) * (4.0*v*rho_dy - 2.0*(u*rho_dx + momy_dy) + momx_dx) * invrho * diffCo1;
    dFdU[3][0][1] -= (1.0/3.0) * (3.0*(3.0*u*u + 4.0*v*v)*rho_dy + 3.0*u*v*rho_dx - 6.0*u*(momx_dy + momy_dx) - 4.0*v*(-momx_dx + 2.0*momy_dy)) * invrho * diffCo1 + 
                                 (-e_dy + (2.0*e*invrho - 3.0*(u*u + v*v))*rho_dy + 2.0*(u*momx_dy + v*momy_dy)) * invrho * diffCo2;

    dFdU[1][1][1] -= -rho_dy * invrho * diffCo1;
    dFdU[2][1][1] -= (2.0/3.0) * rho_dx * invrho * diffCo1;
    dFdU[3][1][1] -= -(1.0/3.0) * (v*rho_dx + 6.0*u*rho_dy - 3.0*(momx_dy + momy_dx)) * invrho * diffCo1 + 
                                  (2.0*u*rho_dy - momx_dy) * invrho * diffCo2;

    dFdU[1][2][1] -= -rho_dx * invrho * diffCo1;
    dFdU[2][2][1] -= -(4.0/3.0) * rho_dy * invrho * diffCo1;
    dFdU[3][2][1] -= -(1.0/3.0) * (u*rho_dx + 8.0*v*rho_dy + 2.0*momx_dx - 4.0*momy_dy) * invrho * diffCo1 + 
                                  (2.0*v*rho_dy - momy_dy) * invrho * diffCo2;

    dFdU[3][3][1] -= -rho_dy * invrho * diffCo2;
  }

  else if (nDims == 3)
  {
    /* Primitive Variables */
    double u = U[1] * invrho;
    double v = U[2] * invrho;
    double w = U[3] * invrho;
    double e = U[4];

    /* Gradients */
    double rho_dx = Q[0][0];
    double momx_dx = Q[1][0];
    double momy_dx = Q[2][0];
    double momz_dx = Q[3][0];
    double e_dx = Q[4][0];
    
    double rho_dy = Q[0][1];
    double momx_dy = Q[1][1];
    double momy_dy = Q[2][1];
    double momz_dy = Q[3][1];
    double e_dy = Q[4][1];

    double rho_dz = Q[0][2];
    double momx_dz = Q[1][2];
    double momy_dz = Q[2][2];
    double momz_dz = Q[3][2];
    double e_dz = Q[4][2];

    /* Set viscous dFdU values in the x-direction */
    dFdU[1][0][0] -= (2.0/3.0) * (4.0*u*rho_dx - 2.0*(momx_dx + v*rho_dy + w*rho_dz) + momy_dy + momz_dz) * invrho * diffCo1;
    dFdU[2][0][0] -= (2.0*(v*rho_dx + u*rho_dy) - (momx_dy + momy_dx)) * invrho * diffCo1;
    dFdU[3][0][0] -= (2.0*(w*rho_dx + u*rho_dz) - (momx_dz + momz_dx)) * invrho * diffCo1;
    dFdU[4][0][0] -= (1.0/3.0) * (3.0*(4.0*u*u + 3.0*(v*v + w*w))*rho_dx + 3.0*(u*v*rho_dy + u*w*rho_dz) 
                                  - 4.0*u*(2.0*momx_dx - momy_dy - momz_dz) - 6.0*(v*(momy_dx + momx_dy) + w*(momz_dx + momx_dz))) * invrho * diffCo1 + 
                                 (-e_dx + (2.0*e*invrho - 3.0*(u*u + v*v + w*w))*rho_dx + 2.0*(u*momx_dx + v*momy_dx + w*momz_dx)) * invrho * diffCo2;

    dFdU[1][1][0] -= -(4.0/3.0) * rho_dx * invrho * diffCo1;
    dFdU[2][1][0] -= -rho_dy * invrho * diffCo1;
    dFdU[3][1][0] -= -rho_dz * invrho * diffCo1;
    dFdU[4][1][0] -= -(1.0/3.0) * (8.0*u*rho_dx + v*rho_dy + w*rho_dz - 4.0*momx_dx + 2.0*(momy_dy + momz_dz)) * invrho * diffCo1 + 
                                  (2.0*u*rho_dx - momx_dx) * invrho * diffCo2;

    dFdU[1][2][0] -= (2.0/3.0) * rho_dy * invrho * diffCo1;
    dFdU[2][2][0] -= -rho_dx * invrho * diffCo1;
    dFdU[4][2][0] -= -(1.0/3.0) * (6.0*v*rho_dx + u*rho_dy - 3.0*(momx_dy + momy_dx)) * invrho * diffCo1 + 
                                  (2.0*v*rho_dx - momy_dx) * invrho * diffCo2;

    dFdU[1][3][0] -= (2.0/3.0) * rho_dz * invrho * diffCo1;
    dFdU[3][3][0] -= -rho_dx * invrho * diffCo1;
    dFdU[4][3][0] -= -(1.0/3.0) * (6.0*w*rho_dx + u*rho_dz - 3.0*(momx_dz + momz_dx)) * invrho * diffCo1 + 
                                  (2.0*w*rho_dx - momz_dx) * invrho * diffCo2;

    dFdU[4][4][0] -= -rho_dx * invrho * diffCo2;

    /* Set viscous dFdU values in the y-direction */
    dFdU[1][0][1] -= (2.0*(u*rho_dy + v*rho_dx) - (momy_dx + momx_dy)) * invrho * diffCo1;
    dFdU[2][0][1] -= (2.0/3.0) * (4.0*v*rho_dy - 2.0*(momy_dy + u*rho_dx + w*rho_dz) + momx_dx + momz_dz) * invrho * diffCo1;
    dFdU[3][0][1] -= (2.0*(w*rho_dy + v*rho_dz) - (momy_dz + momz_dy)) * invrho * diffCo1;
    dFdU[4][0][1] -= (1.0/3.0) * (3.0*(4.0*v*v + 3.0*(u*u + w*w))*rho_dy + 3.0*(v*u*rho_dx + v*w*rho_dz) 
                                  - 4.0*v*(2.0*momy_dy - momx_dx - momz_dz) - 6.0*(u*(momx_dy + momy_dx) + w*(momz_dy + momy_dz))) * invrho * diffCo1 + 
                                 (-e_dy + (2.0*e*invrho - 3.0*(u*u + v*v + w*w))*rho_dy + 2.0*(u*momx_dy + v*momy_dy + w*momz_dy)) * invrho * diffCo2;

    dFdU[1][1][1] -= -rho_dy * invrho * diffCo1;
    dFdU[2][1][1] -= (2.0/3.0) * rho_dx * invrho * diffCo1;
    dFdU[4][1][1] -= -(1.0/3.0) * (6.0*u*rho_dy + v*rho_dx - 3.0*(momx_dy + momy_dx)) * invrho * diffCo1 + 
                                  (2.0*u*rho_dy - momx_dy) * invrho * diffCo2;

    dFdU[1][2][1] -= -rho_dx * invrho * diffCo1;
    dFdU[2][2][1] -= -(4.0/3.0) * rho_dy * invrho * diffCo1;
    dFdU[3][2][1] -= -rho_dz * invrho * diffCo1;
    dFdU[4][2][1] -= -(1.0/3.0) * (8.0*v*rho_dy + u*rho_dx + w*rho_dz - 4.0*momy_dy + 2.0*(momx_dx + momz_dz)) * invrho * diffCo1 + 
                                  (2.0*v*rho_dy - momy_dy) * invrho * diffCo2;

    dFdU[2][3][1] -= (2.0/3.0) * rho_dz * invrho * diffCo1;
    dFdU[3][3][1] -= -rho_dy * invrho * diffCo1;
    dFdU[4][3][1] -= -(1.0/3.0) * (6.0*w*rho_dy + v*rho_dz - 3.0*(momz_dy + momy_dz)) * invrho * diffCo1 + 
                                  (2.0*w*rho_dy - momz_dy) * invrho * diffCo2;

    dFdU[4][4][1] -= -rho_dy * invrho * diffCo2;

    /* Set viscous dFdU values in the z-direction */
    dFdU[1][0][2] -= (2.0*(u*rho_dz + w*rho_dx) - (momz_dx + momx_dz)) * invrho * diffCo1;
    dFdU[2][0][2] -= (2.0*(v*rho_dz + w*rho_dy) - (momz_dy + momy_dz)) * invrho * diffCo1;
    dFdU[3][0][2] -= (2.0/3.0) * (4.0*w*rho_dz - 2.0*(momz_dz + u*rho_dx + v*rho_dy) + momx_dx + momy_dy) * invrho * diffCo1;
    dFdU[4][0][2] -= (1.0/3.0) * (3.0*(4.0*w*w + 3.0*(u*u + v*v))*rho_dz + 3.0*(w*u*rho_dx + w*v*rho_dy) 
                                  - 4.0*w*(2.0*momz_dz - momx_dx - momy_dy) - 6.0*(u*(momx_dz + momz_dx) + v*(momy_dz + momz_dy))) * invrho * diffCo1 + 
                                 (-e_dz + (2.0*e*invrho - 3.0*(u*u + v*v + w*w))*rho_dz + 2.0*(u*momx_dz + v*momy_dz + w*momz_dz)) * invrho * diffCo2;

    dFdU[1][1][2] -= -rho_dz * invrho * diffCo1;
    dFdU[3][1][2] -= (2.0/3.0) * rho_dx * invrho * diffCo1;
    dFdU[4][1][2] -= -(1.0/3.0) * (6.0*u*rho_dz + w*rho_dx - 3.0*(momx_dz + momz_dx)) * invrho * diffCo1 + 
                                  (2.0*u*rho_dz - momx_dz) * invrho * diffCo2;

    dFdU[2][2][2] -= -rho_dz * invrho * diffCo1;
    dFdU[3][2][2] -= (2.0/3.0) * rho_dy * invrho * diffCo1;
    dFdU[4][2][2] -= -(1.0/3.0) * (6.0*v*rho_dz + w*rho_dy - 3.0*(momy_dz + momz_dy)) * invrho * diffCo1 + 
                                  (2.0*v*rho_dz - momy_dz) * invrho * diffCo2;

    dFdU[1][3][2] -= -rho_dx * invrho * diffCo1;
    dFdU[2][3][2] -= -rho_dy * invrho * diffCo1;
    dFdU[3][3][2] -= -(4.0/3.0) * rho_dz * invrho * diffCo1;
    dFdU[4][3][2] -= -(1.0/3.0) * (8.0*w*rho_dz + u*rho_dx + v*rho_dy - 4.0*momz_dz + 2.0*(momx_dx + momy_dy)) * invrho * diffCo1 + 
                                  (2.0*w*rho_dz - momz_dz) * invrho * diffCo2;

    dFdU[4][4][2] -= -rho_dz * invrho * diffCo2;
  }
}

template <size_t nVars, size_t nDims>
#ifdef _GPU
__device__ __forceinline__
#endif
void compute_dFdQvisc_EulerNS(double U[nVars], double dFdQ[nVars][nVars][nDims][nDims], 
    double gamma, double prandtl, double mu)
{
  double invrho = 1.0 / U[0];
  double diffCo1 = mu * invrho;
  double diffCo2 = gamma * mu * invrho / prandtl;

  if (nDims == 2)
  {
    /* Primitive Variables */
    double u = U[1] * invrho;
    double v = U[2] * invrho;
    double e = U[3];

    /* Set viscous dFxdQx values */
    dFdQ[1][0][0][0] = 4.0/3.0 * u * diffCo1;
    dFdQ[2][0][0][0] = v * diffCo1;
    dFdQ[3][0][0][0] = (4.0/3.0 * u*u + v*v) * diffCo1 - (u*u + v*v - e*invrho) * diffCo2;

    dFdQ[1][1][0][0] = -4.0/3.0 * diffCo1;
    dFdQ[3][1][0][0] = -u * (4.0/3.0 * diffCo1 - diffCo2);

    dFdQ[2][2][0][0] = -diffCo1;
    dFdQ[3][2][0][0] = -v * (diffCo1 - diffCo2);

    dFdQ[3][3][0][0] = -diffCo2;

    /* Set viscous dFydQx values */
    dFdQ[1][0][1][0] = v * diffCo1;
    dFdQ[2][0][1][0] = -2.0/3.0 * u * diffCo1;
    dFdQ[3][0][1][0] = 1.0/3.0 * u * v * diffCo1;

    dFdQ[2][1][1][0] = 2.0/3.0 * diffCo1;
    dFdQ[3][1][1][0] = 2.0/3.0 * v * diffCo1;

    dFdQ[1][2][1][0] = -diffCo1;
    dFdQ[3][2][1][0] = -u * diffCo1;

    /* Set viscous dFxdQy values */
    dFdQ[1][0][0][1] = -2.0/3.0 * v * diffCo1;
    dFdQ[2][0][0][1] = u * diffCo1;
    dFdQ[3][0][0][1] = 1.0/3.0 * u * v * diffCo1;

    dFdQ[2][1][0][1] = -diffCo1;
    dFdQ[3][1][0][1] = -v * diffCo1;

    dFdQ[1][2][0][1] = 2.0/3.0 * diffCo1;
    dFdQ[3][2][0][1] = 2.0/3.0 * u * diffCo1;

    /* Set viscous dFydQy values */
    dFdQ[1][0][1][1] = u * diffCo1;
    dFdQ[2][0][1][1] = 4.0/3.0 * v * diffCo1;
    dFdQ[3][0][1][1] = (u*u + 4.0/3.0 * v*v) * diffCo1 - (u*u + v*v - e*invrho) * diffCo2;

    dFdQ[1][1][1][1] = -diffCo1;
    dFdQ[3][1][1][1] = -u * (diffCo1 - diffCo2);

    dFdQ[2][2][1][1] = -4.0/3.0 * diffCo1;
    dFdQ[3][2][1][1] = -v * (4.0/3.0 * diffCo1 - diffCo2);

    dFdQ[3][3][1][1] = -diffCo2;
  }

  else if (nDims == 3)
  {
    /* Primitive Variables */
    double u = U[1] * invrho;
    double v = U[2] * invrho;
    double w = U[3] * invrho;
    double e = U[4];

    /* Set viscous dFxdQx values */
    dFdQ[1][0][0][0] = 4.0/3.0 * u * diffCo1;
    dFdQ[2][0][0][0] = v * diffCo1;
    dFdQ[3][0][0][0] = w * diffCo1;
    dFdQ[4][0][0][0] = (4.0/3.0 * u*u + v*v + w*w) * diffCo1 - (u*u + v*v + w*w - e*invrho) * diffCo2;

    dFdQ[1][1][0][0] = -4.0/3.0 * diffCo1;
    dFdQ[4][1][0][0] = -u * (4.0/3.0 * diffCo1 - diffCo2);

    dFdQ[2][2][0][0] = -diffCo1;
    dFdQ[4][2][0][0] = -v * (diffCo1 - diffCo2);

    dFdQ[3][3][0][0] = -diffCo1;
    dFdQ[4][3][0][0] = -w * (diffCo1 - diffCo2);

    dFdQ[4][4][0][0] = -diffCo2;

    /* Set viscous dFydQx values */
    dFdQ[1][0][1][0] = v * diffCo1;
    dFdQ[2][0][1][0] = -2.0/3.0 * u * diffCo1;
    dFdQ[4][0][1][0] = 1.0/3.0 * u * v * diffCo1;

    dFdQ[2][1][1][0] = 2.0/3.0 * diffCo1;
    dFdQ[4][1][1][0] = 2.0/3.0 * v * diffCo1;

    dFdQ[1][2][1][0] = -diffCo1;
    dFdQ[4][2][1][0] = -u * diffCo1;

    /* Set viscous dFzdQx values */
    dFdQ[1][0][2][0] = w * diffCo1;
    dFdQ[3][0][2][0] = -2.0/3.0 * u * diffCo1;
    dFdQ[4][0][2][0] = 1.0/3.0 * u * w * diffCo1;

    dFdQ[3][1][2][0] = 2.0/3.0 * diffCo1;
    dFdQ[4][1][2][0] = 2.0/3.0 * w * diffCo1;

    dFdQ[1][3][2][0] = -diffCo1;
    dFdQ[4][3][2][0] = -u * diffCo1;

    /* Set viscous dFxdQy values */
    dFdQ[1][0][0][1] = -2.0/3.0 * v * diffCo1;
    dFdQ[2][0][0][1] = u * diffCo1;
    dFdQ[4][0][0][1] = 1.0/3.0 * u * v * diffCo1;

    dFdQ[2][1][0][1] = -diffCo1;
    dFdQ[4][1][0][1] = -v * diffCo1;

    dFdQ[1][2][0][1] = 2.0/3.0 * diffCo1;
    dFdQ[4][2][0][1] = 2.0/3.0 * u * diffCo1;

    /* Set viscous dFydQy values */
    dFdQ[1][0][1][1] = u * diffCo1;
    dFdQ[2][0][1][1] = 4.0/3.0 * v * diffCo1;
    dFdQ[3][0][1][1] = w * diffCo1;
    dFdQ[4][0][1][1] = (u*u + 4.0/3.0 * v*v + w*w) * diffCo1 - (u*u + v*v + w*w - e*invrho) * diffCo2;

    dFdQ[1][1][1][1] = -diffCo1;
    dFdQ[4][1][1][1] = -u * (diffCo1 - diffCo2);

    dFdQ[2][2][1][1] = -4.0/3.0 * diffCo1;
    dFdQ[4][2][1][1] = -v * (4.0/3.0 * diffCo1 - diffCo2);

    dFdQ[3][3][1][1] = -diffCo1;
    dFdQ[4][3][1][1] = -w * (diffCo1 - diffCo2);

    dFdQ[4][4][1][1] = -diffCo2;

    /* Set viscous dFzdQy values */
    dFdQ[2][0][2][1] = w * diffCo1;
    dFdQ[3][0][2][1] = -2.0/3.0 * v * diffCo1;
    dFdQ[4][0][2][1] = 1.0/3.0 * v * w * diffCo1;

    dFdQ[3][2][2][1] = 2.0/3.0 * diffCo1;
    dFdQ[4][2][2][1] = 2.0/3.0 * w * diffCo1;

    dFdQ[2][3][2][1] = -diffCo1;
    dFdQ[4][3][2][1] = -v * diffCo1;

    /* Set viscous dFxdQz values */
    dFdQ[1][0][0][2] = -2.0/3.0 * w * diffCo1;
    dFdQ[3][0][0][2] = u * diffCo1;
    dFdQ[4][0][0][2] = 1.0/3.0 * u * w * diffCo1;

    dFdQ[3][1][0][2] = -diffCo1;
    dFdQ[4][1][0][2] = -w * diffCo1;

    dFdQ[1][3][0][2] = 2.0/3.0 * diffCo1;
    dFdQ[4][3][0][2] = 2.0/3.0 * u * diffCo1;

    /* Set viscous dFydQz values */
    dFdQ[2][0][1][2] = -2.0/3.0 * w * diffCo1;
    dFdQ[3][0][1][2] = v * diffCo1;
    dFdQ[4][0][1][2] = 1.0/3.0 * v * w * diffCo1;

    dFdQ[3][2][1][2] = -diffCo1;
    dFdQ[4][2][1][2] = -w * diffCo1;

    dFdQ[2][3][1][2] = 2.0/3.0 * diffCo1;
    dFdQ[4][3][1][2] = 2.0/3.0 * v * diffCo1;

    /* Set viscous dFzdQz values */
    dFdQ[1][0][2][2] = u * diffCo1;
    dFdQ[2][0][2][2] = v * diffCo1;
    dFdQ[3][0][2][2] = 4.0/3.0 * w * diffCo1;
    dFdQ[4][0][2][2] = (u*u + v*v + 4.0/3.0*w*w) * diffCo1 - (u*u + v*v + w*w - e*invrho) * diffCo2;

    dFdQ[1][1][2][2] = -diffCo1;
    dFdQ[4][1][2][2] = -u * (diffCo1 - diffCo2);

    dFdQ[2][2][2][2] = -diffCo1;
    dFdQ[4][2][2][2] = -v * (diffCo1 - diffCo2);

    dFdQ[3][3][2][2] = -4.0/3.0 * diffCo1;
    dFdQ[4][3][2][2] = -w * (4.0/3.0 * diffCo1 - diffCo2);

    dFdQ[4][4][2][2] = -diffCo2;
  }
}
