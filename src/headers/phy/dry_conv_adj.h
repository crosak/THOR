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
// Description: dry convective adjustment scheme
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


__global__ void dry_conv_adj(double *Pressure_d,    // Pressure [Pa]
                             double *Temperature_d, // Temperature [K]
                             double *profx_Qheat_d,
                             double *pt_d,        // Potential temperature [K]
                             double *Rho_d,       // Density [m^3/kg]
                             double *Cp_d,        // Specific heat capacity [J/kg/K]
                             double *Rd_d,        // Gas constant [J/kg/K]
                             double  Gravit,      // Gravity [m/s^2]
                             double *Altitude_d,  // Altitudes of the layers
                             double *Altitudeh_d, // Altitudes of the interfaces
                             double  time_step,
                             int  conv_adj_iter, // number of iterations of entire algorithm allowed
                             bool soft_adjust,
                             int  num, // Number of columns
                             int  nv) { // Vertical levels
    //
    //  Description: Mixes entropy vertically on statically unstable columns
    //

    int         id = blockIdx.x * blockDim.x + threadIdx.x;

    // stability threshold
    double      stable = 0.0;

    double      ps, psm;

    if (id < num) {
        int  iter   = 0;
        bool repeat = true; //will repeat entire
        while ((repeat == true) && (iter < conv_adj_iter)) {

            psm = Pressure_d[id * nv + 1]
                  + (Pressure_d[id * nv + 1] - Pressure_d[id * nv + 0])
                        / (Altitude_d[1] - Altitude_d[0]) * (-Altitude_d[0] - Altitude_d[1]);
            ps = 0.5 * (Pressure_d[id * nv + 0] + psm);

            // Compute Potential Temperature
            for (int lev = 0; lev < nv; lev++) {
                pt_d[id * nv + lev] = Temperature_d[id * nv + lev]
                                      * pow(ps / Pressure_d[id * nv + lev],
                                            Rd_d[id * nv + lev] / Cp_d[id * nv + lev]);
            }

            bool done_col = false;
            while (done_col == false) { // Unstable  column?
                int top = 0;
                int bot = nv - 1;

                for (int lev = 0; lev < nv - 1; lev++) {
                    // sweep upward, find lowest unstable layer
                    if (pt_d[id * nv + lev + 1] - pt_d[id * nv + lev] < stable) {
                        if (bot > lev)
                            bot = lev;
                    }
                }

                for (int lev = bot; lev < nv - 1; lev++) {
                    // sweep upward from unstable layer, find top
                    if (pt_d[id * nv + lev + 1] - pt_d[id * nv + lev] > stable) {
                        top = lev;
                        break;
                    }
                    else {
                        top = nv - 1;
                    }
                }

                if (bot < nv - 1) {
                    int    extend = 1;
                    double thnew;

                    while (extend == 1) {
                        double h   = 0.0; //Enthalpy;
                        double sum = 0.0;
                        extend     = 0;

                        for (int lev = bot; lev <= top; lev++) {
                            // calc adiabatic pressure, integrate upward for new pot. temp.
                            double rho_g_dz;
                            double pi =
                                pow(Pressure_d[id * nv + lev] / ps,
                                    Rd_d[id * nv + lev]
                                        / Cp_d[id * nv
                                               + lev]); // adiabatic pressure wrt bottom of column
                            // double deltap = pl - pu;
                            rho_g_dz = Rho_d[id * nv + lev] * Gravit
                                       * (Altitudeh_d[lev + 1] - Altitudeh_d[lev]);

                            h   = h + pt_d[id * nv + lev] * pi * rho_g_dz;
                            sum = sum + pi * rho_g_dz;
                        }
                        thnew = h / sum;

                        if (bot > 0) {
                            // repeat if new pot. temp. is less than lower boundary p.t.
                            if ((thnew - pt_d[id * nv + bot - 1]) < stable) {
                                bot    = bot - 1;
                                extend = 1;
                            }
                        }

                        if (top < nv - 1) {
                            // repeat if new pot. temp. is greater p.t. above
                            if ((pt_d[id * nv + top + 1] - thnew) < stable) {
                                top    = top + 1;
                                extend = 1;
                            }
                        }
                    }

                    for (int lev = bot; lev <= top; lev++) {
                        pt_d[id * nv + lev] = thnew; // set new potential temperature
                    }
                }
                else {
                    done_col = true; //no unstable layers
                }
            }

            repeat = false;
            iter += 1;

            if (soft_adjust) {
                double Ttmp, Ptmp;

                for (int lev = 0; lev < nv; lev++) {
                    Ttmp = pt_d[id * nv + lev]
                           * pow(Pressure_d[id * nv + lev] / ps,
                                 Rd_d[id * nv + lev] / Cp_d[id * nv + lev]);
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
                    Temperature_d[id * nv + lev] = pt_d[id * nv + lev]
                                                   * pow(Pressure_d[id * nv + lev] / ps,
                                                         Rd_d[id * nv + lev] / Cp_d[id * nv + lev]);
                    Pressure_d[id * nv + lev] =
                        Temperature_d[id * nv + lev] * Rd_d[id * nv + lev] * Rho_d[id * nv + lev];
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
    }
}
