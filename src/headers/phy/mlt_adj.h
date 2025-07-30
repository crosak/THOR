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
// Current Code Owner: Can AKIN Can.Akin@physik.lmu.de
//
// History:
// Version Date       Comment
// ======= ====       =======
//
////////////////////////////////////////////////////////////////////////
#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

__device__ inline int binary_search(const double *a, double x, int n) {
    int left  = 0;
    int right = n - 1;
    int mid;
    while (left <= right) {
        mid = (left + right) / 2;
        if (x < a[mid]) {
            right = mid - 1;
        }
        else {
            left = mid + 1;
        }
    }
    int index = max(0, right);
    return index;
}

__device__ inline void weno4_q(double  x,
                               double  xim,
                               double  xi,
                               double  xip,
                               double  xipp,
                               double  yim,
                               double  yi,
                               double  yip,
                               double  yipp,
                               double *q2,
                               double *q3) {
    double him = xi - xim;
    double hi  = xip - xi;
    double hip = xipp - xip;

    *q2 = yim * ((x - xi) * (x - xip)) / (him * (him + hi));
    *q2 -= yi * ((x - xim) * (x - xip)) / (him * hi);
    *q2 += yip * ((x - xim) * (x - xi)) / ((him + hi) * hi);

    *q3 = yi * ((x - xip) * (x - xipp)) / (hi * (hi + hip));
    *q3 -= yip * ((x - xi) * (x - xipp)) / (hi * hip);
    *q3 += yipp * ((x - xi) * (x - xip)) / ((hi + hip) * hip);
}

__device__ inline void weno4_B(double  xim,
                               double  xi,
                               double  xip,
                               double  xipp,
                               double  yim,
                               double  yi,
                               double  yip,
                               double  yipp,
                               double *B2,
                               double *B3) {
    double him = xi - xim;
    double hi  = xip - xi;
    double hip = xipp - xip;
    double H   = him + hi + hip;
    double yyim, yyi, yyip, yyipp;

    yyim = -((2.0 * him + hi) * H + him * (him + hi)) / (him * (him + hi) * H) * yim;
    yyim += ((him + hi) * H) / (him * hi * (hi + hip)) * yi;
    yyim -= (him * H) / ((him + hi) * hi * hip) * yip;
    yyim += (him * (him + hi)) / ((hi + hip) * hip * H) * yipp;

    yyi = -(hi * (hi + hip)) / (him * (him + hi) * H) * yim;
    yyi += (hi * (hi + hip) - him * (2.0 * hi + hip)) / (him * hi * (hi + hip)) * yi;
    yyi += (him * (hi + hip)) / ((him + hi) * hi * hip) * yip;
    yyi -= (him * hi) / ((hi + hip) * hip * H) * yipp;

    yyip = (hi * hip) / (him * (him + hi) * H) * yim;
    yyip -= (hip * (him + hi)) / (him * hi * (hi + hip)) * yi;
    yyip += ((him + 2.0 * hi) * hip - (him + hi) * hi) / ((him + hi) * hi * hip) * yip;
    yyip += ((him + hi) * hi) / ((hi + hip) * hip * H) * yipp;

    yyipp = -((hi + hip) * hip) / (him * (him + hi) * H) * yim;
    yyipp += (hip * H) / (him * hi * (hi + hip)) * yi;
    yyipp -= ((hi + hip) * H) / ((him + hi) * hi * hip) * yip;
    yyipp += ((2.0 * hip + hi) * H + hip * (hi + hip)) / ((hi + hip) * hip * H) * yipp;


    double s1   = hi + hip;
    double s2   = him + hi;
    double dsq1 = fabs((yyip - yyi) / hi - (yyi - yyim) / him);
    double dsq2 = fabs((yyipp - yyip) / hip - (yyip - yyi) / hi);
    *B2         = s1 * s1 * dsq1 * dsq1;
    *B3         = s2 * s2 * dsq2 * dsq2;
}

__device__ inline double compute_weno4(int           id,
                                       double        x,
                                       const double *xp,
                                       const double *fp,
                                       int           i,
                                       int           Ngrid,
                                       double        eps,
                                       double       *B2,
                                       double       *B3,
                                       int          *prevB) {
    double y;
    double xim, xi, xip, xipp, yim, yi, yip, yipp;
    double q2, q3, gam2, gam3, al2, al3, om2, om3;

    xi  = xp[i];
    xip = xp[i + 1];
    yi  = fp[id * Ngrid + i];
    yip = fp[id * Ngrid + i + 1];

    // Handle edge cases as per the original Fortran code
    // if (i == 0)
    // {
    //     xim = 0.0;           // Set xim to zero for the lower boundary
    //     xipp = xp[i + 2];
    //     yim = 0.0;           // Set yim to zero for the lower boundary
    //     yipp = fp[id * Ngrid + i + 2];
    // }
    // else if (i == Ngrid - 2)
    // {
    //     xim = xp[i - 1];
    //     xipp = 0.0;          // Set xipp to zero for the upper boundary
    //     yim = fp[id * Ngrid + i - 1];
    //     yipp = 0.0;          // Set yipp to zero for the upper boundary
    // }
    // --- ghost point fix
    if (i == 0) {
        xim  = 2.0 * xp[0] - xp[1];
        yim  = fp[id * Ngrid + 0]; // mirror value
        xipp = xp[2];
        yipp = fp[id * Ngrid + 2];
    }
    else if (i == Ngrid - 3) { // new upper-edge test
        xim  = xp[i - 1];
        yim  = fp[id * Ngrid + i - 1];
        xipp = 2.0 * xp[Ngrid - 1] - xp[Ngrid - 2];
        yipp = fp[id * Ngrid + Ngrid - 1];
    }
    else {
        xim  = xp[i - 1];
        xipp = xp[i + 2];
        yim  = fp[id * Ngrid + i - 1];
        yipp = fp[id * Ngrid + i + 2];
    }

    // Compute q2 and q3 using the weno4_q function
    weno4_q(x, xim, xi, xip, xipp, yim, yi, yip, yipp, &q2, &q3);

    // Determine the interpolated value based on the position
    if (i == 0) {
        y = q3; // Use q3 at the lower boundary
    }
    else if (i == Ngrid - 3) {
        y = q2; // Use q2 at the upper boundary
    }
    else {
        // Recompute B2 and B3 only when i changes
        if (i != *prevB) {
            weno4_B(xim, xi, xip, xipp, yim, yi, yip, yipp, B2, B3);
            *prevB = i;
        }

        // Compute the weights
        gam2 = -(x - xipp) / (xipp - xim);
        gam3 = (x - xim) / (xipp - xim);

        al2 = gam2 / (eps + *B2);
        al3 = gam3 / (eps + *B3);

        om2 = al2 / (al2 + al3);
        om3 = al3 / (al2 + al3);

        // Compute the final interpolated value
        y = om2 * q2 + om3 * q3;
    }

    return y;
}

__device__ inline void interpolate_weno4_kernel(double       *xs,
                                                const double *xp,
                                                const double *fp_column,
                                                double       *result,
                                                const int     nv,
                                                int           num,
                                                bool          use_extrapolate) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < num) {

        int Ngrid = nv;

        double eps = 1.0e-6;
        double B2 = 0.0, B3 = 0.0;
        int    prevB = -1;

        int i;

        for (int lev = 0; lev < nv + 1; lev++) {

            double x = xs[lev];
            // Bottom edge
            if (x < xp[0]) {
                if (!use_extrapolate) {
                    result[id * (nv + 1) + lev] = fp_column[id * nv + 0];
                    continue;
                }
                i = 0;

                // Top edge
            }
            else if (x > xp[Ngrid - 1]) {
                if (!use_extrapolate) {
                    result[id * (nv + 1) + lev] = fp_column[id * nv + Ngrid - 1];
                    continue;
                }
                i = Ngrid - 3;
            }
            else {
                // Normal cells in between
                i = binary_search(xp, x, Ngrid);
            }

            if (i == Ngrid - 1) // x exactly at top layer centre
                i = Ngrid - 2;
            if (i >= Ngrid - 2)
                i = Ngrid - 3; // 4-point stencil fits

            result[id * (nv + 1) + lev] =
                compute_weno4(id, x, xp, fp_column, i, Ngrid, eps, &B2, &B3, &prevB);
        }
    }
}

__device__ inline void bezier_altitude_interpolation(int     id,
                                                         int     nlay,
                                                         int     iter,
                                                         double *xi,
                                                         double *yi,
                                                         double  x,
                                                         double &y) {

    double dx, dx1, dy, dy1;
    double w, yc, t;
    //xc = (xi(1) + xi(2))/2.0_dp ! Control point (no needed here, implicitly included)
    dx  = xi[iter] - xi[iter + 1];
    dx1 = xi[iter - 1] - xi[iter];
    dy  = yi[id * nlay + iter] - yi[id * nlay + iter + 1];
    dy1 = yi[id * nlay + iter - 1] - yi[id * nlay + iter];

    if (x > xi[iter + 1] && x < xi[iter]) {
        // left hand side interpolation
        w = dx1 / (dx + dx1);

        yc = yi[id * nlay + iter] - dx / 2.0 * (w * dy / dx + (1.0 - w) * dy1 / dx1);

        t = (x - xi[iter + 1]) / dx;

        y = pow(1.0 - t, 2) * yi[id * nlay + iter + 1] + 2.0 * t * (1.0 - t) * yc
            + pow(t, 2) * yi[id * nlay + iter];
    }
    else {
        // right hand side interpolation
        w = dx / (dx + dx1);

        yc = yi[id * nlay + iter] + dx1 / 2.0 * (w * dy1 / dx1 + (1.0 - w) * dy / dx);

        t = (x - xi[iter]) / (dx1);

        y = pow(1.0 - t, 2) * yi[id * nlay + iter] + 2.0 * t * (1.0 - t) * yc
            + pow(t, 2) * yi[id * nlay + iter - 1];
    }
}


__global__ void mixing_length_adj(double *Pressure_d,     // Pressure (cell centers) [Pa]
                  double *Temperature_d,  // Temperature (cell centers)[K]
                  double *Temperatureh_d, // Temperature at interfaces (cell edges) [K]
                  double *profx_Qheat_d,
                  double *pt_d,        // Potential temperature [K]
                  double *Rho_d,       // Density [m^3/kg]
                  double *Cp_d,        // Specific heat capacity [J/kg/K]
                  double *Rd_d,        // Gas constant [J/kg/K]
                  double  Gravit,      // Gravity [m/s^2]
                  double  A,           // Planetary radius [m]
                  double *Altitude_d,  // Altitudes of the layers
                  double *Altitudeh_d, // Altitudes of the interfaces
                  double *Kzz_d,       // Eddy diffusion coefficient
                  double *Kzz_ov_d,
                  double *F_conv_d,     // Vertical thermal convective flux [W/m^2]
                  double *F_convh_d,    // Vertical thermal convective flux at interfaces [W/m^2]
                  double *lapse_rate_d, // Lapse rate [K/m]
                  double *tempcolumn_d,
                  double *pcolumn_d,
                  double  mlt_timestep, // Mixing Length Theory time-step [s]
                  double  time_step,    // time step [s]
                  bool    soft_adjust,
                  int     num,          // Number of columns
                  int     nv,           // Vertical levels
                  bool    GravHeightVar)
{
    //
    //  Description:
    //
    // Calculate kappa_ad = Rd_d / Cp_d
    // Calculate gamma_ad = Gravit/Cp_d
    // Interpolate the temperature to the interfaces to calculate the lapse rate over the extent of the cell
    // Calculate gamma = - dT/dz (lapse rate)
    // Calculate the scale height and declare a variable alpha that is ad-hoc, set the default value according to Lee+2023
    // Calculate vertical velocity w
    // Sweep through the atmosphere and trigger an if statement for convective instability
    // Calculate the convective heat flux F_conv = 0.5 * rho * Cp_d * w  * L * (T * gamma - gamma_ad)
    // Interpolate the convective heat flux to the interfaces to calculate the lapse rate over the extent of the cell
    // Calculate the flux derivative across the unstable cells (dF_conv/dz)
    // Calculate the temperature gradient (dT/dt)_mlt = -1/(Cp_d *rho) * (dF_conv/dz)
    // Use the calculated temperature gradient to update the temperature by Tempreature_d = Temperature_d + (dT/dt)_mlt * time_step_mlt
    // Repeat until we reach big time step
    // Pass the temperature tendency by modifying Q_heat (soft adjustment) or by directly updating the T, P, pt, etc. (hard adjustment)

    int id = blockIdx.x * blockDim.x + threadIdx.x;

    // Interpolation variables
    double ps, psm;

    // Constants and parameters
    const double alpha   = 1.0; // MLT scale parameter (Lee+23)
    const double beta    = 2.2; //
    const double Kzz_min = 1e1;
    const double Kzz_max = 1e8;
    double       gamma_ad;                     // Adiabatic lapse rate [K/m]
    double       w_mlt_d, w_mlt_rcb_d, w_ov_d; // Convective velocity [m/s]
    double       scale_height_local_d;         // Scale height for the local conditions
    double       L_d;                          // Characteristic mixing length [m]
    double       dTdz_d;                       // Vertical temperature gradient [K/m] (Lapse rate)
    double       dFdz_d;        // Vertical gradient of the thermal convective flux [W/m^3]
    double       dTempdt_mlt_d; // Temperature tendency due to MLT [K/s]

    // Separate time-stepping for the MLT routine
    double dt;

    if (id < num) {
        // Fill in the temporary array to be used in calculations
        for (int lev = 0; lev < nv; lev++) {
            // Copy initial temperature and pressure arrays
            tempcolumn_d[id * nv + lev] = Temperature_d[id * nv + lev];
            pcolumn_d[id * nv + lev]    = Pressure_d[id * nv + lev];
            // Initialize Kzz arrays to zero
            Kzz_d[id * nv + lev]    = 0.0;
            Kzz_ov_d[id * nv + lev] = 0.0;
        }

        // Calculate the bottom interface pressure through an extrapolation
        if (GravHeightVar) {
            psm = Pressure_d[id * nv + 1]
                  - Rho_d[id * nv + 0] * Gravit * pow(A / (A + Altitude_d[0]), 2)
                        * (-Altitude_d[0] - Altitude_d[1]);
        }
        else {
            psm = Pressure_d[id * nv + 1]
                  - Rho_d[id * nv + 0] * Gravit * (-Altitude_d[0] - Altitude_d[1]);
        }

        ps = 0.5 * (Pressure_d[id * nv + 0] + psm);

        // Initialize iteration properties
        double t_now              = 0.0;
        dt                        = mlt_timestep;
        int  iter                 = 0;
        bool convective_this_step = false; // set true inside ONE sub-step
        bool convective_ever      = false; // set true if triggered at least once
        bool implicit_extrapolate = true;

        // Main sub-timestepping loop //
        while ((t_now < time_step) && iter < 10000) {

            // Adjust time step if it overshoots
            if ((t_now + mlt_timestep >= time_step)) {
                dt = time_step - t_now;
            }

            // Compute Potential Temperature
            for (int lev = 0; lev < nv; lev++) {
                pt_d[id * nv + lev] =
                    tempcolumn_d[id * nv + lev]
                    * pow(ps / pcolumn_d[id * nv + lev], Rd_d[id * nv + lev] / Cp_d[id * nv + lev]);
            }

            // Interpolate temperatures from layers (cell-centered) to levels (interfaces)
            // Bezier Interpolation
            // for (int lev = nv - 1; lev > 0; lev--) {
            //     bezier_altitude_interpolation(
            //         id, nv, lev, Altitude_d, tempcolumn_d, Altitudeh_d[lev], Temperatureh_d[id * (nv + 1) + lev]);
            // }

            // WENO4 interpolation
            interpolate_weno4_kernel(
                Altitudeh_d, Altitude_d, tempcolumn_d, Temperatureh_d, nv, num, implicit_extrapolate);

            if (!implicit_extrapolate) {
                // Linear extrapolation at the lower boundary
                Temperatureh_d[id * (nv + 1) + 0] =
                    tempcolumn_d[id * nv + 0]
                    + (Altitudeh_d[0] - Altitude_d[0])
                          * (Temperatureh_d[id * (nv + 1) + 1] - tempcolumn_d[id * nv + 0])
                          / (Altitudeh_d[1] - Altitude_d[0]);

                // Linear extrapolation at the upper boundary
                Temperatureh_d[id * (nv + 1) + nv] =
                    tempcolumn_d[id * nv + nv - 1]
                    + (Altitudeh_d[nv] - Altitude_d[nv - 1])
                          * (Temperatureh_d[id * (nv + 1) + nv - 1]
                             - tempcolumn_d[id * nv + nv - 1])
                          / (Altitudeh_d[nv - 1] - Altitude_d[nv - 1]);
            }

            // Calculate lapse rate between layers
            for (int lev = 0; lev < nv; lev++) {
                dTdz_d =
                    (Temperatureh_d[id * (nv + 1) + lev + 1] - Temperatureh_d[id * (nv + 1) + lev])
                    / (Altitudeh_d[lev + 1] - Altitudeh_d[lev]);
                lapse_rate_d[id * nv + lev] = -1.0 * dTdz_d; // Γ = -dT/dz
            }

            // Set convection check
            convective_this_step = false;
            // Iterate over all of the levels and check for convective instability
            for (int lev = 0; lev < nv; lev++) {

                // Compute the pressure scale height
                scale_height_local_d = (Rd_d[id * nv + lev] * tempcolumn_d[id * nv + lev]) / Gravit;

                // Mixing length
                L_d = alpha * scale_height_local_d;

                // Calculate adiabatic lapse rate
                gamma_ad = Gravit / Cp_d[id * nv + lev];

                // Sweep upward and check for instability
                if (lapse_rate_d[id * nv + lev] > gamma_ad) {
                    convective_this_step = true;
                    // Calculate the characteristic vertical velocity
                    w_mlt_d = L_d
                              * sqrt(Gravit / tempcolumn_d[id * nv + lev]
                                     * (lapse_rate_d[id * nv + lev] - gamma_ad));
                    // Calculate the convective heat flux (Joyce & Tayar 2023)
                    F_conv_d[id * nv + lev] = 0.5 * Rho_d[id * nv + lev] * Cp_d[id * nv + lev]
                                              * w_mlt_d * L_d
                                              * (lapse_rate_d[id * nv + lev] - gamma_ad);
                }
                else {
                    F_conv_d[id * nv + lev] = 0.0;
                    w_mlt_d                 = 0.0;
                }

                // Update Kzz running total
                Kzz_d[id * nv + lev] += w_mlt_d * L_d;
            }

            // Check if convective instability got triggered
            if (!convective_this_step) {
                iter += 1;
                break;
            }

            // If code reaches this far convection occured
            convective_ever = true;

            // Interpolate the vertical convective thermal flux
            // for (int lev = nv - 1; lev > 0; lev--) {
            //     bezier_altitude_interpolation(
            //         id, nv, lev, Altitude_d, F_conv_d, Altitudeh_d[lev], F_convh_d[id * (nv + 1) + lev]);
            // }

            interpolate_weno4_kernel(
                Altitudeh_d, Altitude_d, F_conv_d, F_convh_d, nv, num, implicit_extrapolate);

            if (!implicit_extrapolate) {
                // Linear interapolation to the lower boundary
                // F_convh_d[id * (nv + 1) + 0] = F_conv_d[id * nv + 0] + (Altitudeh_d[0] - Altitude_d[0])
                //                                     * (F_convh_d[id * (nv + 1) + 1] - F_conv_d[id * nv + 0]) / (Altitudeh_d[1] - Altitude_d[0]);

                // Linear interapolation to the upper boundary
                // F_convh_d[id * (nv + 1) + nv] = F_conv_d[id * nv + nv - 1] + (Altitudeh_d[nv] - Altitude_d[nv - 1])
                //                                     * (F_convh_d[id * (nv + 1) + nv - 1] - F_conv_d[id * nv + nv - 1]) / (Altitudeh_d[nv-1] - Altitude_d[nv-1]);

                // Set the edges to zero
                F_convh_d[id * (nv + 1) + 0]  = 0.0;
                F_convh_d[id * (nv + 1) + nv] = 0.0;
            }


            for (int lev = 0; lev < nv; lev++) {
                // Calculate the flux derivative (dF_conv/dz)
                dFdz_d = (F_convh_d[id * (nv + 1) + lev + 1] - F_convh_d[id * (nv + 1) + lev])
                         / (Altitudeh_d[lev + 1] - Altitudeh_d[lev]);
                // Calculate the temperature gradient
                dTempdt_mlt_d = -1.0 / (Cp_d[id * nv + lev] * Rho_d[id * nv + lev]) * dFdz_d;
                // Update the temperature in a sub-timestep approach using a smaller timestep than the dynamical timestep
                tempcolumn_d[id * nv + lev] = tempcolumn_d[id * nv + lev] + dTempdt_mlt_d * dt;
                // Update the pressure with the updated temperature
                pcolumn_d[id * nv + lev] =
                    tempcolumn_d[id * nv + lev] * Rd_d[id * nv + lev] * Rho_d[id * nv + lev];
            }

            // Update the iteration counter & time step
            iter += 1;
            t_now += dt;
        }

        // If no correction happened set Kzz to minimum value
        if (!convective_ever) {
            for (int lev = 0; lev < nv; ++lev)
                Kzz_d[id * nv + lev] = Kzz_min;

            // Early return since we can skip the rest of the calculations
            return;
        }

        // Find the final averaged K_zz value
        for (int lev = 0; lev < nv; lev++) {
            Kzz_d[id * nv + lev] = Kzz_d[id * nv + lev] / (double)iter;
        }

        // Calculate the overshoot component //
        // Find the RCB
        int krcb    = 0;
        w_mlt_rcb_d = 1e-30;
        for (int lev = 0; lev < nv; lev++) {
            if (F_conv_d[id * nv + lev] > 0.0) {
                krcb = lev;
            }
            else if (krcb != 0) {
                // Keep in mind this might cause errors if you
                // removed the early return case handling above
                krcb = lev - 1;
                scale_height_local_d =
                    (Rd_d[id * nv + krcb] * tempcolumn_d[id * nv + krcb]) / Gravit;
                L_d         = alpha * scale_height_local_d;
                gamma_ad    = Gravit / Cp_d[id * nv + krcb];
                w_mlt_rcb_d = L_d
                              * sqrt(Gravit / tempcolumn_d[id * nv + krcb]
                                     * (lapse_rate_d[id * nv + krcb] - gamma_ad));
                break;
            }
        }

        // Overshoot component
        for (int lev = 0; lev < nv; lev++) {
            if (lev <= krcb) {
                // In convective region - do not add overshoot
            }
            else {
                // In overshoot region, add overshoot component
                w_ov_d = exp(
                    log(w_mlt_rcb_d)
                    - beta * fmax(0.0, log(pcolumn_d[id * nv + krcb] / pcolumn_d[id * nv + lev])));
                scale_height_local_d = (Rd_d[id * nv + lev] * tempcolumn_d[id * nv + lev]) / Gravit;
                L_d                  = alpha * scale_height_local_d;
                Kzz_ov_d[id * nv + lev] = w_ov_d * L_d;
                if (Kzz_ov_d[id * nv + lev] < Kzz_min) {
                    break;
                }
            }
        }

        for (int lev = 0; lev < nv; lev++) {
            // Make sure Kzz is above minimum value
            Kzz_d[id * nv + lev] = fmax(Kzz_d[id * nv + lev] + Kzz_ov_d[id * nv + lev], Kzz_min);

            // Make sure Kzz is smaller than the maximum value
            Kzz_d[id * nv + lev] = fmin(Kzz_d[id * nv + lev], Kzz_max);
        }

        // Soft adjust the results by only modifying the Qheat term using the calculated temperature
        if (soft_adjust) {
            double Ttmp, Ptmp;

            for (int lev = 0; lev < nv; lev++) {
                Ttmp = tempcolumn_d[id * nv + lev];
                Ptmp = Ttmp * Rd_d[id * nv + lev] * Rho_d[id * nv + lev];
                //reset pt value to beginning of time step
                pt_d[id * nv + lev] = Temperature_d[id * nv + lev]
                                      * pow(Pressure_d[id * nv + lev] / ps,
                                            -Rd_d[id * nv + lev] / Cp_d[id * nv + lev]);
                profx_Qheat_d[id * nv + lev] += (Cp_d[id * nv + lev] - Rd_d[id * nv + lev])
                                                / Rd_d[id * nv + lev]
                                                * (Ptmp - Pressure_d[id * nv + lev]) / time_step;
            }
        }
        // Hard adjust the pressure and the pot. temperature directly using the calculated temperature
        else {
            for (int lev = 0; lev < nv; lev++) {
                Temperature_d[id * nv + lev] = tempcolumn_d[id * nv + lev];
                Pressure_d[id * nv + lev] =
                    Temperature_d[id * nv + lev] * Rd_d[id * nv + lev] * Rho_d[id * nv + lev];

                pt_d[id * nv + lev] = Temperature_d[id * nv + lev]
                                      * pow(Pressure_d[id * nv + lev] / ps,
                                            -Rd_d[id * nv + lev] / Cp_d[id * nv + lev]);
            }
        }
    }
}


// ----------------------------------------------------------------------------------------------//
// Experimental optimized MLT implementation 

__device__ inline double compute_weno4_local_parallel(double        x,
                                                      const double *xp,
                                                      const double *fp,
                                                      int           i,
                                                      int           Ngrid,
                                                      double        eps,
                                                      double       *B2,
                                                      double       *B3,
                                                      int          *prevB)
{
    double y;
    double xim, xi, xip, xipp, yim, yi, yip, yipp;
    double q2, q3, gam2, gam3, al2, al3, om2, om3;

    xi  = xp[i];
    xip = xp[i + 1];
    yi  = fp[i];
    yip = fp[i + 1];

    // --- ghost point fix
    if (i == 0) {
        xim  = 2.0 * xp[0] - xp[1];
        yim  = fp[0]; // mirror value
        xipp = xp[2];
        yipp = fp[2];
    }
    else if (i == Ngrid - 3) { // new upper-edge test
        xim  = xp[i - 1];
        yim  = fp[i - 1];
        xipp = 2.0 * xp[Ngrid - 1] - xp[Ngrid - 2];
        yipp = fp[Ngrid - 1];
    }
    else {
        xim  = xp[i - 1];
        xipp = xp[i + 2];
        yim  = fp[i - 1];
        yipp = fp[i + 2];
    }

    // Compute q2 and q3 using the weno4_q function
    weno4_q(x, xim, xi, xip, xipp, yim, yi, yip, yipp, &q2, &q3);

    // Determine the interpolated value based on the position
    if (i == 0) {
        y = q3; // Use q3 at the lower boundary
    }
    else if (i == Ngrid - 3) {
        y = q2; // Use q2 at the upper boundary
    }
    else {
        // Recompute B2 and B3 only when i changes
        if (i != *prevB) {
            weno4_B(xim, xi, xip, xipp, yim, yi, yip, yipp, B2, B3);
            *prevB = i;
        }

        // Compute the weights
        gam2 = -(x - xipp) / (xipp - xim);
        gam3 = (x - xim) / (xipp - xim);

        al2 = gam2 / (eps + *B2);
        al3 = gam3 / (eps + *B3);

        om2 = al2 / (al2 + al3);
        om3 = al3 / (al2 + al3);

        // Compute the final interpolated value
        y = om2 * q2 + om3 * q3;
    }

    return y;
}


__device__ inline void weno4_interface(const double *xs,
                                       const double *xp,
                                       const double *fp,
                                       double       *result,
                                       int           lev,
                                       int           nv,
                                       bool          use_extrapolate)
{
    // Altitudeh_d[lev]
    double x = xs[lev];                 
    int Ngrid = nv;
    int i;

    double eps, B2, B3;  
    int    prevB;
    
    eps = 1e-6;
    B2 = 0.0 , B3 = 0.0;
    prevB = -1;

    // Bottom edge
    if (x < xp[0]) {
        if (!use_extrapolate) {
            result[lev] = fp[0];
            return;
        }
        i = 0;
    }
    // Top edge
    else if (x > xp[Ngrid-1]) {
        if (!use_extrapolate) {
            result[lev] = fp[Ngrid-1];
            return;
        }
        i = Ngrid - 3;
    }
    // Normal cells in between
    else {
        i = binary_search(xp, x, Ngrid);
    }                      

    if (i == Ngrid - 1) {
        i = Ngrid - 2;
    }   
    if (i >= Ngrid - 2){
        i = Ngrid - 3;
    } 

    double val = compute_weno4_local_parallel(x, xp, fp, i, Ngrid, eps, &B2, &B3, &prevB);
    result[lev] = val;
}


__global__ void mixing_length_adj_parallel(double *Pressure_d, 
                                           double *Temperature_d, 
                                           double *profx_Qheat_d,  
                                           double *pt_d,           
                                           double *Rho_d,          
                                           double *Cp_d,           
                                           double *Rd_d,           
                                           double  Gravit,
                                           double  A,
                                           double *Altitude_d,    
                                           double *Altitudeh_d,    
                                           double *Kzz_d,       
                                           double  mlt_timestep,
                                           double  time_step,
                                           bool    soft_adjust,
                                           int     num,
                                           int     nv,
                                           bool    GravHeightVar)
{
    const int id  = blockIdx.x;
    const int lev = threadIdx.x;       // vertical level handled by this thread
    const int threads_per_block = blockDim.x;

    // Thread safety
    if (id >= num){
        return;
    } 

    // Interpolation variables
    double psm;
    
    // Physics variables
    double  gamma_ad;               // Adiabatic lapse rate [K/m]
    double  w_mlt, w_mlt_rcb, w_ov; // Convective velocity [m/s]
    double  scale_height_local;     // Scale height for the local conditions
    double  L;                      // Characteristic mixing length [m]
    double  dTdz;                   // Vertical temperature gradient [K/m] (Lapse rate)
    double  dFdz;                   // Vertical gradient of the thermal convective flux [W/m^3]
    double  dTdt_mlt;               // Temperature tendency due to MLT [K/s]


    // Constants and parameters
    const double alpha   = 1.0; // MLT scale parameter (Lee+23)
    const double beta    = 2.2; 
    const double Kzz_min = 1e1;
    const double Kzz_max = 1e8;


    // Shared memory arrays
    extern __shared__ double sh[];
    double* temperature_sh  = &sh[ 0 * threads_per_block]; 
    double* temperatureh_sh = &sh[ 1 * threads_per_block];
    double* pressure_sh     = &sh[ 2 * threads_per_block]; 
    double* rho_sh          = &sh[ 3 * threads_per_block];
    double* Cp_sh           = &sh[ 4 * threads_per_block];
    double* Rd_sh           = &sh[ 5 * threads_per_block];
    double* altitude_sh     = &sh[ 6 * threads_per_block]; 
    double* altitudeh_sh    = &sh[ 7 * threads_per_block];
    double* f_conv_sh       = &sh[ 8 * threads_per_block]; 
    double* f_convh_sh      = &sh[ 9 * threads_per_block]; 
    double* lapse_rate_sh   = &sh[ 10 * threads_per_block]; 
    double* kzz_sh          = &sh[ 11 * threads_per_block]; 
    double* kzzov_sh        = &sh[ 12 * threads_per_block]; 

    // extra scalars
    __shared__ double ps_sh;
    // __shared__ volatile int convective_any;     // set each sub-step
    // __shared__ volatile int convective_ever_sh; // latched once any sub-step convects

    // Load column into shared arrays
    if (lev < nv) {
        temperature_sh[lev]  = Temperature_d[id * nv + lev];
        temperatureh_sh[lev] = 0.0;
        pressure_sh[lev]     = Pressure_d[id * nv + lev];
        rho_sh[lev]          = Rho_d[id * nv + lev];
        Cp_sh[lev]           = Cp_d[id * nv + lev];
        Rd_sh[lev]           = Rd_d[id * nv + lev];
        altitude_sh[lev]     = Altitude_d[lev];
        altitudeh_sh[lev]    = Altitudeh_d[lev];
        f_conv_sh[lev]       = 0.0;
        f_convh_sh[lev]      = 0.0;
        lapse_rate_sh[lev]   = 0.0;
        kzz_sh[lev]          = 0.0;
        kzzov_sh[lev]        = 0.0;
    }
    __syncthreads();

    // Initialize interface array edges
    if (lev == nv) {
        temperatureh_sh[lev] = 0.0;
        altitudeh_sh[lev]    = Altitudeh_d[lev];
        f_convh_sh[lev]      = 0.0;
    }
    
    // Calculate the bottom interface pressure through an extrapolation
    if (lev == 0){
        if (GravHeightVar) {
            psm = pressure_sh[1] - rho_sh[0] * Gravit * pow(A / (A + altitude_sh[0]), 2) * (-altitude_sh[0] - altitude_sh[1]);
        }
        else {
            psm = pressure_sh[1] - rho_sh[0] * Gravit * (-altitude_sh[0] - altitude_sh[1]);
        }

        ps_sh = 0.5 * (pressure_sh[0] + psm);
    }

    __syncthreads();

    const double ps = ps_sh;

    // if (lev == 0) {
    //     convective_ever_sh = 0; // nothing yet
    // }
    // __syncthreads();

    // Initialize iteration properties
    double t_now              = 0.0;
    double dt                 = mlt_timestep;
    int  iter                 = 0;
    bool implicit_extrapolate = true;

    // Main sub-timestepping loop
    while ((t_now < time_step) && iter < 10000) {
        
        // Adjust time step if it overshoots
        if ((t_now + dt >= time_step)) {
            dt = time_step - t_now;
        }

        __syncthreads();

        // WENO4 interpolation
        if (lev <= nv) {
            weno4_interface(altitudeh_sh, altitude_sh, temperature_sh, temperatureh_sh, lev, nv, implicit_extrapolate);
        }
           
        __syncthreads();

        if (!implicit_extrapolate && lev == 0) {
            // Linear extrapolation at the lower boundary
            temperatureh_sh[0] = temperature_sh[0] + (altitudeh_sh[0] - altitude_sh[0])
                                * (temperatureh_sh[1] - temperature_sh[0])
                                / (altitudeh_sh[1] - altitude_sh[0]);

            // Linear extrapolation at the upper boundary
            temperatureh_sh[nv] = temperature_sh[nv - 1] + (altitudeh_sh[nv] - altitude_sh[nv - 1])
                        * (temperatureh_sh[nv - 1] - temperature_sh[nv - 1])
                        / (altitudeh_sh[nv - 1] - altitude_sh[nv - 1]);
        }

        __syncthreads();
        
        // Calculate lapse rate between layers
        if (lev < nv){
            dTdz = (temperatureh_sh[lev + 1] - temperatureh_sh[lev]) / (altitudeh_sh[lev+1] - altitudeh_sh[lev]);
            lapse_rate_sh[lev] = -1.0 * dTdz;
        }

        __syncthreads();

        // Set convection check
        // if (lev == 0){
        //     convective_any = 0;
        // } 

        __syncthreads();

        // Go over all of the levels and check for convective instability
        if (lev < nv){
            
            // Compute the pressure scale height
            scale_height_local = (Rd_sh[lev] * temperature_sh[lev]) / Gravit;
            
            // Mixing length
            L = alpha * scale_height_local;

            // Calculate adiabatic lapse rate
            gamma_ad = Gravit / Cp_sh[lev];

            // Check for convective instabilities
            if (lapse_rate_sh[lev] > gamma_ad) {
                // atomicOr(&convective_any, 1);

                // Calculate the characteristic vertical velocity
                w_mlt = L * sqrt(Gravit / temperature_sh[lev] * (lapse_rate_sh[lev] - gamma_ad));

                // Calculate the convective heat flux (Joyce & Tayar 2023)
                f_conv_sh[lev] = 0.5 * rho_sh[lev] * Cp_sh[lev] * w_mlt * L * (lapse_rate_sh[lev] - gamma_ad);
            }
            else {
                
                w_mlt = 0.0;
                f_conv_sh[lev] = 0.0;
            }

            // Update Kzz running total
            kzz_sh[lev] += w_mlt * L;
        }

        __syncthreads();
        
        // if (lev == 0 && convective_any) {
        //     convective_ever_sh = 1;   // once set, stays set
        // }
        // __syncthreads();

        // Check if convective instability got triggered
        // if (convective_any == 0) {
        //     ++iter;
        //     break; // same as scalar: exit sub-step loop early, no convective_ever
        // }

        // Interpolate the vertical convective thermal flux
        if (lev <= nv) {
            weno4_interface(altitudeh_sh, altitude_sh, f_conv_sh, f_convh_sh, lev, nv, implicit_extrapolate);
        }

        __syncthreads();

        if (!implicit_extrapolate && lev == 0) {
            // Linear interapolation to the lower boundary
            // f_convh_sh[0] = f_conv_sh[0] + (altitudeh_sh[0] - altitude_sh[0])
            //              * (f_convh_sh[1] - f_conv_sh[0]) / (altitudeh_sh[1] - altitude_sh[0]);

            // Linear interapolation to the upper boundary
            // f_convh_sh[nv] = f_conv_sh[nv - 1] + (altitudeh_sh[nv] - altitude_sh[nv - 1])
            //              * (f_convh_sh[nv - 1] - f_conv_sh[nv - 1]) / (altitudeh_sh[nv-1] - Altitude_sh[nv-1]);

            // Set the edges to zero
            f_convh_sh[0]  = 0.0;
            f_convh_sh[nv] = 0.0;
        }

        __syncthreads();

        if (lev < nv){
            
            // Calculate the flux derivative (dF_conv/dz)
            dFdz = (f_convh_sh[lev + 1] - f_convh_sh[lev]) / (altitudeh_sh[lev+1] - altitudeh_sh[lev]);

            // Calculate the temperature gradient
            dTdt_mlt = -1.0 / (Cp_sh[lev] * rho_sh[lev]) * dFdz;

            // Update the temperature in a sub-timestep approach using a smaller timestep than the dynamical timestep
            temperature_sh[lev] += dTdt_mlt * dt;

            // Update the pressure with the updated temperature
            pressure_sh[lev] = temperature_sh[lev] * Rd_sh[lev] * rho_sh[lev];
        }

        __syncthreads();
        
        // Update the iteration counter & time step
        t_now += dt;
        iter++;
    }

    __syncthreads();

    // If no correction happened set K_zz to minimum value
    // if (convective_ever_sh == 0) {
    //     if (lev < nv) {
    //         // write out immediately
    //         Kzz_d[id * nv + lev] = Kzz_min;
    //     }
    //     __syncthreads();
    //     // Early exit: whole block returns
    //     return;
    // }

    // __syncthreads(); 

    // Find the final averaged K_zz value
    if (lev < nv){
        kzz_sh[lev] = kzz_sh[lev] / (double)iter;
    }

    __syncthreads();

    __shared__ int    krcb_sh;
    __shared__ double w_mlt_rcb_sh;

    if (lev == 0) {
        // Walk upward
        int k = 0;
        while (k < nv) {

            // Skip stable layers
            while (k < nv && f_conv_sh[k] <= 0.0) {
                ++k; 
            }
            // It should be impossible to hit this?
            if (k >= nv) {
                break;
            }

            // Inside a convective block
            while (k < nv && f_conv_sh[k] > 0.0) {
                ++k;
            }

            // Last convective level = current RCB
            krcb_sh = k - 1;

            // Mixing velocity at this RCB
            w_mlt_rcb = 1e-30;
            if (krcb_sh < nv) {
                scale_height_local = (Rd_sh[krcb_sh] * temperature_sh[krcb_sh]) / Gravit;
                L        = alpha * scale_height_local;
                gamma_ad = Gravit / Cp_sh[krcb_sh];
                w_mlt_rcb = L * sqrt(fmax(0.0,
                                Gravit / temperature_sh[krcb_sh] *
                            (lapse_rate_sh[krcb_sh] - gamma_ad)));
            }
            w_mlt_rcb_sh = w_mlt_rcb;

            // Overshoot above this RCB until next convective block or top
            
            while (k < nv && f_conv_sh[k] <= 0.0) {
                double kov = 0.0;
                w_ov = exp( log(w_mlt_rcb_sh) -
                            beta * fmax(0.0,
                                log(pressure_sh[krcb_sh] /
                                    pressure_sh[k])) );

                scale_height_local = (Rd_sh[k] * temperature_sh[k]) / Gravit;
                L   = alpha * scale_height_local;
                kov = w_ov * L;

                if (kov < Kzz_min) { kov = 0.0; }

                kzzov_sh[k] = kov;
                // advance upward
                ++k;                              
            }

            // Loop continues from the level that ended the overshoot loop
            // (either next convective layer or nv)
        }
    }

    __syncthreads();

    // Combine Kzz + overshoot, clamp, update global array
    if (lev < nv) {
        // Make sure Kzz is above minimum value
        kzz_sh[lev] = fmax(kzz_sh[lev] + kzzov_sh[lev], Kzz_min);

        // Make sure Kzz is smaller than the maximum value
        Kzz_d[id * nv + lev] = fmin(kzz_sh[lev], Kzz_max);
    }
    
    __syncthreads();

    // Update global arrays to be passed to the dynamical core
    if (lev < nv) {
        // Soft adjust the results by only modifying the Qheat term using the calculated temperature
        if (soft_adjust) {
            double Ptmp, Pold;
            Ptmp = temperature_sh[lev] * Rd_sh[lev] * rho_sh[lev];
            Pold = Pressure_d[id * nv + lev];
            profx_Qheat_d[id * nv + lev] += (Cp_sh[lev] - Rd_sh[lev]) / Rd_sh[lev] * (Ptmp - Pold) / time_step;
        
        // Hard adjust the pressure and the pot. temperature directly using the calculated temperature
        } else {
            Temperature_d[id*nv + lev] = temperature_sh[lev];
            Pressure_d[id*nv + lev]    = temperature_sh[lev] * Rd_sh[lev] * rho_sh[lev];
            pt_d[id*nv + lev] = temperature_sh[lev] * pow(Pressure_d[id*nv + lev] / ps, -Rd_sh[lev]/Cp_sh[lev]);
        }
    }
}