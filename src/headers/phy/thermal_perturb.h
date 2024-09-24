// ================================================================================
// This file is part of THOR.
//
//     THOR is free software : you can redistribute it and / or modify
//     it under the terms of the GNU General Public License as published by
//     the Free Software Foundation, either version 3 of the License, or
//     (at your option) any later version.
//
//     THOR is distributed in the hope that it will be useful,
//     but WITHOUT ANY WARRANTY; without even the implied warranty of
//     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the
//     GNU General Public License for more details.
//
//     You find a copy of the GNU General Public License in the main
//     THOR directory under <license.txt>.If not, see
//     <http://www.gnu.org/licenses/>.
// =================================================================================
//
//
//
//
// Description: Introducing random thermal perturbations at the radiative-convective
//              boundary. This is an implementation of the Showman, Tan & Zhang (2019)
//              scheme, first implemented into Exo-FMS by Elsie Lee. Adapted by Can
//              Akin & Leonardos Gkouvelis for THOR.
//
// Known limitations: None
//
// Known issues: None.
//
//
// If you use this code please cite the following reference:
//
//       [1] Mendonca, J.M., Grimm, S.L., Grosheintz, L., & Heng, K., ApJ, 829, 115, 2016
//
// Current Code Owners: Joao Mendonca (joao.mendonca@space.dtu.dk)
//                      Russell Deitrick (russell.deitrick@csh.unibe.ch)
//                      Urs Schroffenegger (urs.schroffenegger@csh.unibe.ch)
//
// History:
// Version Date       Comment
// ======= ====       =======
//
//
////////////////////////////////////////////////////////////////////////

#include <math.h>
// Will check whether all of these libraries are necessary.
#include <cmath>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// Function to initialize the random number generator state
__device__ void set_seed(curandState* state, int id) {
    curand_init(1234, id, 0, &state[id]);
}

// Device function for generating random numbers
__device__ double ran2(curandState* state) {
    return curand_uniform_double(state);
}
// Device function to compute normalized associated Legendre functions
__device__ void LPMN_NORM(int mmax,
                          int nmax,
                          double x,
                          double* PM_d,
                          int id){

    // Calculate the starting index for the current grid point
    int startIndex = id * (mmax + 1) * (nmax + 1);

    // Declare local variables
    double ls, xq;

    // Initialize PM array to zero for the current grid point
    for (int m = 0; m <= mmax; m++) {
        for (int n = 0; n <= nmax; n++) {
            PM_d[startIndex + m * (nmax + 1) + n] = 0.0;
        }
    }

    // Set the base value for the Legendre function
    PM_d[startIndex] = 0.5 * sqrt(2.0);

    // Special case when x is exactly 1.0
    if (fabs(x) == 1.0) {
        for (int n = 1; n <= nmax; n++) {
            PM_d[startIndex + n] = sqrt(n + 0.5) * pow(x, n);
        }
        return;
    }

    // Compute ls and xq values
    ls = (fabs(x) > 1.0) ? -1.0 : 1.0;
    xq = sqrt(ls * (1.0 - x * x));

    // Compute the diagonal terms of the Legendre function matrix
    for (int m = 1; m <= mmax; m++) {
        PM_d[startIndex + m * (nmax + 1) + m] = ls * sqrt((2.0 * m + 1.0) / (2.0 * m)) * xq * PM_d[startIndex + (m - 1) * (nmax + 1) + (m - 1)];
    }

    // Compute the sub-diagonal terms of the Legendre function matrix
    for (int m = 0; m <= mmax; m++) {
        if (m + 1 <= nmax) {
            PM_d[startIndex + m * (nmax + 1) + (m + 1)] = sqrt(2.0 * m + 3.0) * x * PM_d[startIndex + m * (nmax + 1) + m];
        }
    }

    // Compute the off-diagonal terms of the Legendre function matrix
    for (int m = 0; m <= mmax; m++) {
        for (int n = m + 2; n <= nmax; n++) {
            PM_d[startIndex + m * (nmax + 1) + n] =
                sqrt((2.0 * n + 1.0) * (2.0 * n - 1.0) / ((n - m) * (n + m))) * x * PM_d[startIndex + m * (nmax + 1) + (n - 1)] -
                sqrt((2.0 * n + 1.0) * (n - m - 1.0) * (n + m - 1.0) / ((2.0 * n - 3.0) * (n - m) * (n + m))) * PM_d[startIndex + m * (nmax + 1) + (n - 2)];
        }
    }
}


// Kernel function for thermal perturbation calculation
__global__ void thermal_perturb(double *Pressure_d,    // Pressure [Pa]
                                double *Pressureh_d,   // Mid-point pressure [Pa] or is it interface pressure?
                                double *Temperature_d, // Temperature [K]
                                double *profx_Qheat_d,
                                double *pt_d,          // Potential temperature [K]
                                double *Rho_d,         // Density [m^3/kg]
                                double *Cp_d,          // Specific heat capacity [J/kg/K]
                                double *Rd_d,          // Gas constant [J/kg/K]
                                double  Gravit,        // Gravity [m/s^2]
                                double *Altitude_d,    // Altitudes of the layers
                                double *Altitudeh_d,   // Altitudes of the interfaces
                                double *lonlat_d,
                                double  time_step,
                                double  A,
                                // thermal_perturb working variables 
                                double* thermpert_d,
                                double* bforce_d,      // Thermal forcing
                                double* bturb_d,       // Turbulent perturbation field
                                double* PM_d,          // Normalized associated Legendre functions
                                // thermal_perturb parameters
                                // int Nspecmodes,        // Number of spectral modes (might be unnecessary?)
                                int mforce,            // Order of the assoc. Legendre functions
                                int nforce,            // Degree of the assoc. Legendre functions
                                int delta_n,            
                                int mmax,            
                                int nmax,             
                                int nrforctop,      // Number of forcing top layers
                                double t_storm,        // Storm timescale [s]
                                double t_amp,          // Forcing amplitude [?]
                                double p_rcb,          // Pressure of the radiative-convective boundary [Pa]
                                int nburn,          // Number of burn-in iterations
                                curandState* state,
                                bool first_call,
                                bool soft_adjust,
                                bool GravHeightVar,
                                int  num,              // Number of columns
                                int  nv                // Vertical levels
                                ) {
    
    
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    // Declare and initialize local variables
    const double pi = atan(1.0) * 4;
    double r;
    double forc_vert_profile;

    // Interpolation variables
    double ps, psm;
    double pp, ptop;
    double xi, xip, xim, a, b;
    
    // Calculate decay factor
    r = 1.0 - time_step / t_storm;

    if (id < num) {
        
        double lat_d   = lonlat_d[id * 2 + 1];
        double lon_d   = lonlat_d[id * 2];

        // Calculate pressure at the interfaces
        for (int lev = 0; lev <= nv; lev++) {
            if (lev == 0) {
                // Extrapolate to lower boundary
                if (GravHeightVar) {
                    psm = Pressure_d[id * nv + 1]
                          - Rho_d[id * nv + 0] * Gravit * pow(A / (A + Altitude_d[0]), 2)
                                * (-Altitude_d[0] - Altitude_d[1]);
                }
                else {
                    psm = Pressure_d[id * nv + 1]
                          - Rho_d[id * nv + 0] * Gravit * (-Altitude_d[0] - Altitude_d[1]);
                }
                ps                             = 0.5 * (Pressure_d[id * nv + 0] + psm);
                Pressureh_d[id * (nv + 1) + 0] = ps;
            }
            else if (lev == nv) {
                // Extrapolate to top boundary
                if (GravHeightVar) {
                    pp =
                        Pressure_d[id * nv + nv - 2]
                        - Rho_d[id * nv + nv - 1] * Gravit
                              * pow(A / (A + Altitude_d[nv - 1]), 2)
                              * (2 * Altitudeh_d[nv] - Altitude_d[nv - 1] - Altitude_d[nv - 2]);
                }
                else {
                    pp =
                        Pressure_d[id * nv + nv - 2]
                        - Rho_d[id * nv + nv - 1] * Gravit
                              * (2 * Altitudeh_d[nv] - Altitude_d[nv - 1] - Altitude_d[nv - 2]);
                }
                if (pp < 0)
                    pp = 0; //prevents pressure from going negative
                ptop                             = 0.5 * (Pressure_d[id * nv + nv - 1] + pp);
                Pressureh_d[id * (nv + 1) + lev] = ptop;
            }
            else {
                // Interpolation between layers
                xi  = Altitudeh_d[lev];
                xim = Altitude_d[lev - 1];
                xip = Altitude_d[lev];
                a   = (xi - xip) / (xim - xip);
                b   = (xi - xim) / (xip - xim);
                Pressureh_d[id * (nv + 1) + lev] =
                    Pressure_d[id * nv + lev - 1] * a + Pressure_d[id * nv + lev] * b;
            }
        }

        // Perform initialization and burn-in on the first call 
        // This is a Markov chain, that is why it needs some iterations to generate the desired pattern
        if (first_call) {
            // Initialize the random number generator seed
            set_seed(state, id);

            // Initialize and compute the associated Legendre functions for the current grid point
            double slat = sin(lat_d); 

            // Call the LPMN_NORM function for the current grid point
            LPMN_NORM(mmax, nmax, slat, PM_d, id);

            // Start the burn-in iterations for initializing turbulent perturbations
            for (int i = 0; i < nburn; i++) {
                // Set the initial turbulent perturbations to zero
                for (int lev = 0; lev < nv; lev++) {
                    bturb_d[id * nv + lev] = 0.0;
                }

                // Calculate turbulent perturbations using the Legendre functions
                for (int n = nforce; n < nforce + delta_n; n++) {
                    for (int m = 1; m <= n; m++) {
                        double rphase = ran2(&state[id]) * 2.0 * pi;
                        int pm_index = id * (mmax + 1) * (nmax + 1) + m * (nmax + 1) + n;
                        for (int lev = 0; lev < nv; lev++) {
                            bturb_d[id * nv + lev] += PM_d[pm_index] * cos(m * (lon_d + rphase));
                        }
                    }
                }

                // Update the thermal forcing
                for (int lev = 0; lev < nv; lev++) {
                    bforce_d[id * nv + lev] = r * bforce_d[id * nv + lev] + sqrt(1.0 - r * r) * (t_amp * bturb_d[id * nv + lev]);
                }
            }
        }
    
        // Reset bturb array to zero
        for (int lev = 0; lev < nv; lev++) {
            bturb_d[id * nv + lev] = 0.0;
        }

        // Calculate new turbulent perturbations
        for (int n = nforce; n < nforce + delta_n; n++) {
            for (int m = 1; m <= n; m++) {
                double rphase = ran2(&state[id]) * 2.0 * pi;
                int pm_index = id * (mmax + 1) * (nmax + 1) + m * (nmax + 1) + n;
                for (int lev = 0; lev < nv; lev++) {
                    bturb_d[id * nv + lev] += PM_d[pm_index] * cos(m * (lon_d + rphase));        
                }
            }
        }

        // Update the thermal forcing
        for (int lev = 0; lev < nv; lev++) {
            bforce_d[id * nv + lev] = r * bforce_d[id * nv + lev] + sqrt(1.0 - r * r) * (t_amp * bturb_d[id * nv + lev]);
        }

         // Calculate thermal perturbation
        for (int lev = 0; lev < nv; lev++) {
            double pressure = Pressureh_d[id * nv + lev];
            double thet_over_t = 1.0;  // Placeholder, should use a relevant formula

            // Compute the vertical forcing profile based on pressure levels
            if (pressure <= p_rcb) {
                forc_vert_profile = pow(pressure / p_rcb, 2);
            } else {
                forc_vert_profile = pow(p_rcb / pressure, 2);
            }

            // Apply the thermal perturbation conditionally
            if (pressure <= p_rcb * 7.389 || pressure >= p_rcb / 7.389) {
                thermpert_d[id * nv + lev] = thet_over_t * forc_vert_profile * bforce_d[id * nv + lev];
                // if (id % 100 == 0 && lev == 0) { // Print only for every 100th thread and the first level
                //     printf("Perturbations at point (%.2f, %.2f): %.2e\n", lat_d, lon_d, thermpert_d[id * nv + lev]);
                // }                
            } else {
                thermpert_d[id * nv + lev] = 0.0;
            }
        }
        
        // Apply the updates 
        if (soft_adjust) {
            double Ttmp, Ptmp;

            for (int lev = 0; lev < nv; lev++) {
                Ttmp = Temperature_d[id * nv + lev] + thermpert_d[id * nv + lev];
                Ptmp = Ttmp * Rd_d[id * nv + lev] * Rho_d[id * nv + lev];

                profx_Qheat_d[id * nv + lev] +=
                    (Cp_d[id * nv + lev] - Rd_d[id * nv + lev]) / Rd_d[id * nv + lev]
                    * (Ptmp - Pressure_d[id * nv + lev]) / time_step;
                //does not repeat
            }
        }
        // Set the adjusted temperature and pressure directly rather than passing tendencies with Q_heat
        else {
            for (int lev = 0; lev < nv; lev++) {
                Temperature_d[id * nv + lev] = Temperature_d[id * nv + lev] + thermpert_d[id * nv + lev];
                Pressure_d[id * nv + lev] =
                    Temperature_d[id * nv + lev] * Rd_d[id * nv + lev] * Rho_d[id * nv + lev];

                pt_d[id * nv + lev] = Temperature_d[id * nv + lev]
                                      * pow(Pressure_d[id * nv + lev] / ps,
                                            -Rd_d[id * nv + lev] / Cp_d[id * nv + lev]);
            }
        }

    }
}

