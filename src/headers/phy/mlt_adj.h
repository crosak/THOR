// ==============================================================================
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
// ==============================================================================
//
//
//
//
// Description: mixing length theory adjustment scheme
//
//
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
// 2.0     30/11/2018 Released version (RD & US)
// 1.0     16/08/2017 Released version  (JM)
//
////////////////////////////////////////////////////////////////////////

#include <math.h>


__global__ void mixing_length_adj(double *Pressure_d,  // Pressure [Pa]
                                 double *Pressureh_d, // Mid-point pressure [Pa]
                                 double *dT_conv_d,
                                 double *Temperature_d, // Temperature [K]
                                 double *profx_Qheat_d,
                                 double *pt_d,          // Potential temperature [K]
                                 double *Rho_d,         // Density [m^3/kg]
                                 double *Cp_d,          // Specific heat capacity [J/kg/K]
                                 double *Rd_d,          // Gas constant [J/kg/K]
                                 double  Gravit,        // Gravity [m/s^2]
                                 double *Altitude_d,    // Altitudes of the layers
                                 double *Altitudeh_d,   // Altitudes of the interfaces
                                 double  time_step,     // time step [s]
                                 bool    soft_adjust,
                                 int     num, // Number of columns
                                 int     nv)      // Vertical levels

{
    //
    //  Description: 
    //
    // Keep the interpolation/extrapolation schemes of Russell+Pascal
        // Calculate kappa_ad = Rd_d / Cp_d 
        // Calculate gamma = - dT/dz ``
        // Calculate the scale height and declare a variable alpha that is ad-hoc, set the default value according to Lee+2023
        // Calculate vertical velocity w 
        // Sweep through the atmosphere and trigger an if statement for convective instability
        // Calculate the convective heat flux F_conv = 0.5 * rho * Cp_d * w * Temperature_d * alpha * H * (gamma - gamma_ad)
        // Calculate the flux derivative across the instability zone (dF_conv/dz)
        // Calculate the temperature gradient (dT/dt)_mlt = -1/(Cp_d *rho) * (dF_conv/dz)
        // Use the calculated temperature gradient to update the temperature by Tempreature_d = Temperature_d + (dT/dt)_mlt * time_step_mlt
        // Repeat until we reach big time step``

    int id = blockIdx.x * blockDim.x + threadIdx.x;

    double ps, psm;
    double pp, ptop;
    double alpha=0.1; // MLT constant, set to 0.1 for now (Lee+23) 
    double scale_height, w_mlt, L_mlt, F_conv;
    double gamma_ad = Gravit/Cp_d; // Replaces the stability threshold
    double mlt_timestep = 0.5; // Mixing Length Theory time-step in seconds
    int  mlt_adj_iter = time_step / mlt_timestep; // number of iterations of entire algorithm allowed
    double stable = 0.0; // stability threshold

    double xi, xip, xim, a, b;

    if (id < num) {

        int  iter   = 0;
        bool repeat = true; //will repeat entire
        while ((repeat == true) && (iter < mlt_adj_iter)) {
            // for (iter = 0; iter < ITERMAX; iter++) {
            // calculate pressure at the interfaces
            for (int lev = 0; lev <= nv; lev++) {
                if (lev == 0) {
                    // extrapolate to lower boundary
                    if (GravHeightVar) {
                        psm = Pressure_d[id * nv + 1]
                              - Rho_d[id * nv + 0] * Gravit * pow(A / (A + Altitude_d[0]), 2)
                                    * (-Altitude_d[0] - Altitude_d[1]);
                        // psm = Pressure_d[id * nv + 1]
                        //       - Rho_d[id * nv + 0] * Gravit * (-Altitude_d[0] - Altitude_d[1]);
                    }
                    else {
                        psm = Pressure_d[id * nv + 1]
                              - Rho_d[id * nv + 0] * Gravit * (-Altitude_d[0] - Altitude_d[1]);
                    }
                    ps                             = 0.5 * (Pressure_d[id * nv + 0] + psm);
                    Pressureh_d[id * (nv + 1) + 0] = ps;
                }
                else if (lev == nv) {
                    // extrapolate to top boundary
                    if (GravHeightVar) {
                        pp =
                            Pressure_d[id * nv + nv - 2]
                            - Rho_d[id * nv + nv - 1] * Gravit
                                  * pow(A / (A + Altitude_d[nv - 1]), 2)
                                  * (2 * Altitudeh_d[nv] - Altitude_d[nv - 1] - Altitude_d[nv - 2]);
                        // pp =
                        //     Pressure_d[id * nv + nv - 2]
                        //     - Rho_d[id * nv + nv - 1] * Gravit
                        //           * (2 * Altitudeh_d[nv] - Altitude_d[nv - 1] - Altitude_d[nv - 2]);
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
                    // interpolation between layers
                    xi  = Altitudeh_d[lev];
                    xim = Altitude_d[lev - 1];
                    xip = Altitude_d[lev];
                    a   = (xi - xip) / (xim - xip);
                    b   = (xi - xim) / (xip - xim);
                    Pressureh_d[id * (nv + 1) + lev] =
                    Pressure_d[id * nv + lev - 1] * a + Pressure_d[id * nv + lev] * b;
                }
            }

            // Compute the pressure scale height (Assumes hydrostatic equilibrium <- Will change later)
            for (int lev = 0; lev < nv; lev++) {
                scale_height[id * nv + lev] = (Rd_d * Temperature_d[id * nv + lev]) / Gravit ;
                L[id * nv + lev] = alpha * scale_height[id * nv + lev];
            }

            // Compute the temperature gradient
            for (int lev = 1; lev < nv; lev++) {
                dT_mlt_d[id * nv + lev] = (Temperature_d[id * (nv + 1) + lev] - Temperature_d[id * (nv + 1) + lev - 1]) / ( Altitudeh_d[lev] - Altitude_d[lev-1]);
            }

            // Iterate over all of the levels and check for convective instability
            for (int lev = 0; lev < nv - 1; lev++) {
                // Sweep upward and check for instability
                if (pt_d[id * nv + lev + 1] - pt_d[id * nv + lev] < stable) {
                    // Calculate the characteristic vertical velocity
                    w_mlt[id * nv + lev] = L * sqrt(Gravit/Temperature_d[id * nv + lev] * (dT_mlt_d[id * nv + lev] - gamma_ad));

                    // Calculate the convective heat flux
                    F_conv[id * nv + lev] = 0.5 * Rho_d[id * nv + lev] * Cp_d * w * Temperature_d[id * nv + lev] * alpha * scale_height * (dT_mlt_d[id * nv + lev] - gamma_ad);
                }
                else{
                    F_conv[id * nv + lev] = 0.0;
                }
                    
            }     

            // // Calculate the flux derivative (dF_conv/dz)
            for (int lev = 0; lev < nv - 1; lev++) {

                // Sweep upward and check for instability
                if (pt_d[id * nv + lev + 1] - pt_d[id * nv + lev] < stable) {

                        dFdz[id * nv + lev] =  (F_conv[id * (nv + 1) + lev - 1]- F_conv[id * (nv + 1) + lev - 1]) / (Altitudeh_d[lev] - Altitude_d[lev-1]);

                        // Calculate the temperature gradient
                        dTempdt_mlt[id * nv + lev] = -1/(Cp_d * Rho_d[id * nv + lev]) * dFdz[id * nv + lev]; 
                }
                else {
                    
                    dFdz[id * nv + lev] = 0.0
                    dTempdt_mlt[id * nv + lev] = 0.0
                }

            }

            repeat = false;
            iter += 1;
            if (soft_adjust) {
                double Ttmp, Ptmp;
                // Update the temperature in a sub-timestep approach using a smaller timestep than the dynamical timestep
                for (int lev = 0; lev < nv; lev++) {
                    Ttmp = Temperature_d[id * nv + lev] + dTempdt_mlt[id * nv + lev] * mlt_timestep;
                    Ptmp = Ttmp * Rd_d[id * nv + lev] * Rho_d[id * nv + lev];
                    //reset pt value to beginning of time step
                    pt_d[id * nv + lev] = Temperature_d[id * nv + lev]
                                          * pow(Pressure_d[id * nv + lev] / ps,
                                                -Rd_d[id * nv + lev] / Cp_d[id * nv + lev]);

                    profx_Qheat_d[id * nv + lev] +=
                        (Cp_d[id * nv + lev] - Rd_d[id * nv + lev]) / Rd_d[id * nv + lev]
                        * (Ptmp - Pressure_d[id * nv + lev]) / time_step;
                    //does not repeat
                }
            }
            // Compute Temperature & pressure from potential temperature
            else {
                for (int lev = 0; lev < nv; lev++) {
                    Temperature_d[id * nv + lev] = Temperature_d[id * nv + lev] + dTempdt_mlt[id * nv + lev] * mlt_timestep;
                    Pressure_d[id * nv + lev] = Temperature_d[id * nv + lev] * Rd_d[id * nv + lev] * Rho_d[id * nv + lev];
                    //check pt again
                    pt_d[id * nv + lev] = Temperature_d[id * nv + lev]
                                          * pow(Pressure_d[id * nv + lev] / ps,
                                                -Rd_d[id * nv + lev] / Cp_d[id * nv + lev]);
                    if (lev > 0) {
                        if (pt_d[id * nv + lev] - pt_d[id * nv + lev - 1] < stable)
                            repeat = true;
                    }
                }
            }
            
            

        }
        //printf("id = %d, iter = %d\n", id, iter);
    }
}
