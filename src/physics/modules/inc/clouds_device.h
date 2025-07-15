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
// ESP -  Exoclimes Simulation Platform. (version 2.5)
//
// Description: Adaptation of E. Lee's mini_cloud module into THOR.
//
// Method: Cloud physics module
//
//
// Known limitations: - Runs on a single GPU.
//
// Known issues: None
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
// 1.0     17/12/2024   (CA)
//
////////////////////////////////////////////////////////////////////////
#include <cuda_runtime.h>

__host__ __device__ double p_vap_sp(int species_id, double T) {
    double p_vap = 0.0;
    double Tc;
    // Conversion constants (from Fortran to SI)
    // Fortran defines:
    //   bar = 1.0e6 dyn/cm², atm = 1.01325e6 dyn/cm², pa = 10.0 dyn/cm², mmHg = 1333.22387415 dyn/cm².
    // 1 dyn/cm² = 0.1 Pa, so:
    const double ATM_TO_PA  = 1.01325e6 * 0.1; // 1.01325e5 Pa
    const double BAR_TO_PA  = 1.0e6 * 0.1;     // 1.0e5 Pa
    const double DYNE_TO_PA = 0.1;             // 1 dyn/cm² = 0.1 Pa

    switch (species_id) {
        case 0: // "C"
            // Fortran: p_vap = 10^(-41523.0/T + 10.609) * atm
            p_vap = pow(10.0, -41523.0 / T + 10.609) * ATM_TO_PA;
            break;
        case 1: // "TiO2"
            // Fortran: p_vap = exp(-7.70443e4/T + 40.3144 - 2.59140e-3*T + 6.02422e-7*T^2 - 6.86899e-11*T^3)
            // (No multiplier specified → value is in dyn/cm²)
            p_vap = exp(-7.70443e4 / T + 40.3144 - 2.59140e-3 * T + 6.02422e-7 * T * T
                        - 6.86899e-11 * T * T * T)
                    * DYNE_TO_PA;
            break;
        case 2: // "Al2O3"
            // Fortran: p_vap = exp(-73503.0/T + 22.01) * atm
            p_vap = exp(-73503.0 / T + 22.01) * ATM_TO_PA;
            break;
        case 3: // "Fe"
            // Fortran:
            //   if (T > 1800) then p_vap = exp(9.86 - 37120.0/T)*bar
            //   else             p_vap = exp(15.71 - 47664.0/T)*bar
            if (T > 1800.0) {
                p_vap = exp(9.86 - 37120.0 / T) * BAR_TO_PA;
            }
            else {
                p_vap = exp(15.71 - 47664.0 / T) * BAR_TO_PA;
            }
            break;
        case 4: // "Mg2SiO4"
            // Fortran: p_vap = exp(-62279.0/T + 20.944) * atm
            p_vap = exp(-62279.0 / T + 20.944) * ATM_TO_PA;
            break;
        case 5: // "MgSiO3" or "MgSiO3_amorph"
            // Fortran: p_vap = exp(-58663.0/T + 25.37) * bar
            p_vap = exp(-58663.0 / T + 25.37) * BAR_TO_PA;
            break;
        case 6: // "SiO2" or "SiO2_amorph"
            // Fortran: p_vap = exp(-7.28086e4/T + 36.5312 - 2.56109e-4*T - 5.24980e-7*T^2 + 1.53343e-10*T^3)*bar
            p_vap = exp(-7.28086e4 / T + 36.5312 - 2.56109e-4 * T - 5.24980e-7 * T * T
                        + 1.53343e-10 * T * T * T)
                    * BAR_TO_PA;
            break;
        case 7: // "SiO"
            // Fortran: p_vap = exp(-49520.0/T + 32.52)
            p_vap = exp(-49520.0 / T + 32.52) * DYNE_TO_PA;
            break;
        case 8: // "Cr"
            // Fortran: p_vap = exp(-4.78455e4/T + 32.2423 - 5.28710e-4*T - 6.17347e-8*T^2 + 2.88469e-12*T^3)*bar
            p_vap = exp(-4.78455e4 / T + 32.2423 - 5.28710e-4 * T - 6.17347e-8 * T * T
                        + 2.88469e-12 * T * T * T)
                    * BAR_TO_PA;
            break;
        case 9: // "MnS"
            // Fortran: p_vap = 10^(11.532 - 23810.0/T) * bar
            p_vap = pow(10.0, 11.532 - 23810.0 / T) * BAR_TO_PA;
            break;
        case 10: // "Na2S"
            // Fortran: p_vap = 10^(8.550 - 13889.0/T) * bar
            p_vap = pow(10.0, 8.550 - 13889.0 / T) * BAR_TO_PA;
            break;
        case 11: // "ZnS"
            // Fortran: p_vap = exp(-4.75507888e4/T + 36.6993865 - 2.49490016e-3*T + 7.29116854e-7*T^2 - 1.12734453e-10*T^3)*bar
            p_vap = exp(-4.75507888e4 / T + 36.6993865 - 2.49490016e-3 * T + 7.29116854e-7 * T * T
                        - 1.12734453e-10 * T * T * T)
                    * BAR_TO_PA;
            break;
        case 12: // "KCl"
            // Fortran: p_vap = exp(-2.69250e4/T + 33.9574 - 2.04903e-3*T - 2.83957e-7*T^2 + 1.82974e-10*T^3)
            // (No multiplier, so result is in dyn/cm²)
            p_vap = exp(-2.69250e4 / T + 33.9574 - 2.04903e-3 * T - 2.83957e-7 * T * T
                        + 1.82974e-10 * T * T * T)
                    * DYNE_TO_PA;
            break;
        case 13: // "NaCl"
            // Fortran: p_vap = exp(-2.79146e4/T + 34.6023 - 3.11287e3*T + 5.30965e-7*T^2 - 2.59584e-12*T^3)
            // (No multiplier → in dyn/cm²)
            p_vap = exp(-2.79146e4 / T + 34.6023 - 3.11287e3 * T + 5.30965e-7 * T * T
                        - 2.59584e-12 * T * T * T)
                    * DYNE_TO_PA;
            break;
        case 14: // "NH4Cl"
            // Fortran: p_vap = 10^(7.0220 - 4302.0/T) * bar
            p_vap = pow(10.0, 7.0220 - 4302.0 / T) * BAR_TO_PA;
            break;
        case 15: // "H2O"
            // Fortran logic:
            //   if (T > 1048) then p_vap = 6.0e8 dyn/cm²
            //   else if (T < 273.16) then p_vap = 6111.5 dyn/cm² * exp((23.036*Tc - Tc^2/333.7)/(Tc+279.82))
            //   else p_vap = 6112.1 dyn/cm² * exp((18.729*Tc - Tc^2/227.3)/(Tc+257.87))
            Tc = T - 273.15;
            if (T > 1048.0) {
                p_vap = 6.0e8 * DYNE_TO_PA;
            }
            else if (T < 273.16) {
                p_vap =
                    6111.5 * DYNE_TO_PA * exp((23.036 * Tc - (Tc * Tc) / 333.7) / (Tc + 279.82));
            }
            else {
                p_vap =
                    6112.1 * DYNE_TO_PA * exp((18.729 * Tc - (Tc * Tc) / 227.3) / (Tc + 257.87));
            }
            break;
        case 16: // "NH3"
            // Fortran: p_vap = exp(10.53 - 2161.0/T - 86596.0/T^2) * bar
            p_vap = exp(10.53 - 2161.0 / T - 86596.0 / (T * T)) * BAR_TO_PA;
            break;
        case 17: // "CH4"
            // Fortran:
            //   if (T < 0.5) then p_vap = 10^(3.9895 - 443.028/(0.5-0.49)) * bar
            //   else p_vap = 10^(3.9895 - 443.028/(T-0.49)) * bar
            if (T < 0.5) {
                p_vap = pow(10.0, 3.9895 - 443.028 / (0.5 - 0.49)) * BAR_TO_PA;
            }
            else {
                p_vap = pow(10.0, 3.9895 - 443.028 / (T - 0.49)) * BAR_TO_PA;
            }
            break;
        case 18: // "NH4SH"
            // Fortran: p_vap = 10^(7.8974 - 2409.4/T) * bar
            p_vap = pow(10.0, 7.8974 - 2409.4 / T) * BAR_TO_PA;
            break;
        case 19: // "H2S"
            // Fortran:
            //   if (T < 30) then p_vap = 10^(4.43681 - 829.439/(30-25.412)) * bar
            //   else if (T < 212.8) then p_vap = 10^(4.43681 - 829.439/(T-25.412)) * bar
            //   else p_vap = 10^(4.52887 - 958.587/(T-0.539)) * bar
            if (T < 30.0) {
                p_vap = pow(10.0, 4.43681 - 829.439 / (30.0 - 25.412)) * BAR_TO_PA;
            }
            else if (T < 212.8) {
                p_vap = pow(10.0, 4.43681 - 829.439 / (T - 25.412)) * BAR_TO_PA;
            }
            else {
                p_vap = pow(10.0, 4.52887 - 958.587 / (T - 0.539)) * BAR_TO_PA;
            }
            break;
        case 20: // "S2"
            // Fortran:
            //   if (T < 413) then p_vap = exp(27.0 - 18500.0/T) * bar
            //   else p_vap = exp(16.1 - 14000.0/T) * bar
            if (T < 413.0) {
                p_vap = exp(27.0 - 18500.0 / T) * BAR_TO_PA;
            }
            else {
                p_vap = exp(16.1 - 14000.0 / T) * BAR_TO_PA;
            }
            break;
        case 21: // "S8"
            // Fortran:
            //   if (T < 413) then p_vap = exp(20.0 - 11800.0/T) * bar
            //   else p_vap = exp(9.6 - 7510.0/T) * bar
            if (T < 413.0) {
                p_vap = exp(20.0 - 11800.0 / T) * BAR_TO_PA;
            }
            else {
                p_vap = exp(9.6 - 7510.0 / T) * BAR_TO_PA;
            }
            break;
        default:
            printf("Saturation: species not found for ID %d\n", species_id);
            break;
    }
    return p_vap;
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

__device__ inline void dqdt(double *Pressure_d, // Pressure (cell centers) [Pa]
                     double *Cp_d,       // Specific heat capacity [J/kg/K]
                     double *y,          // Work array for tracers
                     double *f,          // Tracer time derivatives
                     double  p_vap,      // Saturation vapour pressure
                     double  tau_deep_d, // Deep vapour replenishment timescale [s]
                     double  q_v_deep,   // Deep replenishment rate in VMR
                     double  q_s,        // Equilibrium vapour pressure fraction
                     double  dt,         // Integration time
                     int     id,         // Column id
                     int     icloud,     // Current cloud species
                     int     ntr_cloud,  // Number of cloud tracers
                     int     lev,        // Current vertical level
                     int     nv,         // Vertical levels
                     bool    deep_flag) {

    // Current convention:
    // y[0] = q_v, y[1] = q_c, y[2] = T_latent

    // Declare variables
    double sat, diff, replenishment_rate;

    // Indexing variables
    const int STRIDE_ID  = nv * ntr_cloud;
    const int STRIDE_LEV = ntr_cloud;
    int       base_var   = 2 * icloud;

    // Assign vapor and condensate indices
    int idx_v = id * STRIDE_ID + lev * STRIDE_LEV + base_var + 0;
    int idx_c = id * STRIDE_ID + lev * STRIDE_LEV + base_var + 1;
    // int idx_T = ;

    // Calculate the supersaturation ratio
    sat = fmax((y[idx_v] * Pressure_d[id * nv + lev]) / p_vap, 1e-99);

    // Calculate dqdt given the supersaturation ratio
    if (sat < 0.99) {
        // Evaporate from q_c
        f[idx_v] = fmin(q_s - y[idx_v], y[idx_c]) / dt;
    }
    else if (sat > 1.01) {
        // Condense q_v toward the saturation ratio
        f[idx_v] = -(y[idx_v] - q_s) / dt;
    }
    else {
        f[idx_v] = 0.0;
    }

    f[idx_c] = -1.0 * f[idx_v];

    // Add replenishment to lower boundary at the tau_deep rate
    if (deep_flag) {
        diff               = (y[idx_v] - q_v_deep);
        replenishment_rate = diff / tau_deep_d;
        f[idx_v]           = f[idx_v] - replenishment_rate;
        f[idx_c]           = 0.0;
    }

    // Change in atmospheric temperature due to latent heat
    // f[idx_T] = L_heat/ Cp_d * f[idx_c];
}

__global__ void mini_cloud(double *Pressure_d,       // Pressure (cell centers) [Pa]
                           double *Temperature_d,    // Tem perature (cell centers)[K]
                           double *Rho_d,            // Density [kg/m3]
                           double *Cp_d,             // Specific heat capacity [J/kg/K]
                           double *Rd_d,             // Gas constant [J/kg/K]
                           double  Gravit,           // Gravity [m/s2]
                           double  tau_deep_d,       // Replenishment timescale [s]
                           // double  Kzz_deep,      // Deep replenisment rate for Kzz
                           double *tracer_cloud_d,
                           double *q_v_deep_vmr_d,   // Deep vapor mixing ratio
                           // bool    latent_flag,
                           double *vf_d,             // Settling velocity [m/s]
                           double  tau_chem_d,       // Equilibrium cloud timescale [s]
                           double *rho_pd_d,         // Particle density [kg/m3]
                           double  rm_d,             // Median particle size [m]
                           int    *active_species_d, // Integer identifiers for active cloud species
                           double *mol_w_sp_d,       // Molecular weight of species [kg/mol]
                           double *y,                // Work arrays
                           double *f,
                           double  time_step, // time step [s]
                           int     ntr_cloud, // Number of cloud tracers
                           int     num,       // Number of columns
                           bool    GravHeightVar

) {
    // Get column id
    int id     = blockIdx.x * blockDim.x + threadIdx.x;
    int nv     = gridDim.y;
    int lev    = blockIdx.y;
    int icloud = blockIdx.z;

    // Universal gas constant
    const double R_UNIV_th = 8.31446261815324; // universal gas constant in J / ( K mol )
    // const double K_B_th      = 1.38064852e-23;         // Boltzmann constant in J / K
    // const double AMU_th      = 1.6605390666e-27;       // atomic mass unit in kg

    // Temporary variables for computations
    bool   deep_flag;
    double mu, eps, p_vap, q_v_deep;
    // double scale_height_local_d, L_heat;
    double q_v, q_c, q_s;
    double dt, t_now;

    // Local working variables for the Bogacki–Shampine (order 3) Runge-Kutta method
    double k1_v, k1_c, k2_v, k2_c, k3_v, k3_c;
    double y_old_v, y_old_c, y3_v, y3_c, y2_v, y2_c;
    double y_in_v, y_in_c;

    // Error calculation & adaptive time stepping variables
    bool accept;
    double err_max_ratio, factor, tol_c, tol_v;
    const double poww  = 0.2;
    const double safe  = 0.9;
    const double a_tol = 1e-30;
    const double r_tol = 5e-2;
    const double dt_min = 1e-2;

    if (id < num) {
        // Indexing variables
        const int STRIDE_ID  = nv * ntr_cloud;
        const int STRIDE_LEV = ntr_cloud;
        int       base_var   = 2 * icloud;

        // Assign vapor and condensate indices to variables for consistency
        int idx_v = id * STRIDE_ID + lev * STRIDE_LEV + base_var + 0;
        int idx_c = id * STRIDE_ID + lev * STRIDE_LEV + base_var + 1;
        // int idx_T = ;

        // Retrieve vapor and condensate mass-mixing ratios (The rho terms are there to ensure we are working with MMR values)
        q_v = tracer_cloud_d[idx_v] / Rho_d[id * nv + lev];
        q_c = tracer_cloud_d[idx_c] / Rho_d[id * nv + lev];

        // if (q_v < 0.0 || isnan(q_v)){
        //     printf("Negative tracer concentrations passed from outside!! \n");
        //     printf("q_v_vmr = %.3e | Time-step: %f \n", q_v, time_step);
        // }
        // if (q_c < 0.0 || isnan(q_c)){
        //     printf("Negative tracer concentrations passed from outside!! \n");
        //     printf("q_c_vmr = %.3e | Time-step: %f \n", q_c, time_step);
        // }

        // Compute the pressure scale height [m]
        // scale_height_local_d = (Rd_d[id * nv + lev] * Temperature_d[id * nv + lev]) / Gravit ;

        // Deep replenishment timescale tau_deep [s]
        // tau_deep_d = pow(scale_height_local_d, 2) / Kzz_deep;

        // Molar mass [kg/mol]
        mu = R_UNIV_th / Rd_d[id * nv + lev];

        // Conversion factor between MMR and VMR for cloud species. MMR --> VMR
        eps = mol_w_sp_d[icloud] / mu;

        // Deep vapour reservoir VMR
        q_v_deep = q_v_deep_vmr_d[icloud];

        // Saturation vapour pressure
        p_vap = p_vap_sp(active_species_d[icloud], Temperature_d[id * nv + lev]);

        // Latent heat release from sublimation/vapourisation
        // if (latent_flag){
        //     L_heat = l_heat_sp(active_species_d[icloud], Temperature_d[id * nv + lev]); // Add a device function for l_heat_sp
        // }
        // else{
        //     L_heat = 0.0;
        // }
        // L_heat = 0.0;

        // Equilibrium vapour VMR
        q_s = p_vap / Pressure_d[id * nv + lev];
        q_s = fmin(q_s, 1.0);

        // Calculate timescale with latent heat lag term
        // dt = tau_chem_d * (1.0 + (L_heat * L_heat * q_s * mol_w_sp_d[icloud])
        //     / (Cp_d[id * nv + lev] * R_UNIV_th * Temperature_d[id * nv + lev] * Temperature_d[id * nv + lev]));
        dt = tau_chem_d;     

        // Check if we're in the "deep" region
        if (lev == 0) {
            deep_flag = true;
        }
        else {
            deep_flag = false;
        }

        // Ensure that tracers are non-negative
        q_v = fmax(q_v, 1e-30);
        q_c = fmax(q_c, 1e-30);

        // Initial conditions - convert to VMR from MMR
        y[idx_v] = q_v / eps;
        y[idx_c] = q_c / eps;

        // // Initial atmospheric temperature
        // y[idx_T] = Temperature_d[id * nv + lev];

        // Copy the initial value before time iteration
        double yc_v, yc_c;
        yc_v = y[idx_v];
        yc_c = y[idx_c];

        // Start time integration
        t_now                 = 0.0;
        int    n_it           = 0;
        double cloud_timestep = dt;
        while ((t_now < time_step) && n_it < 10000) {

            if (t_now + cloud_timestep >= time_step) {
                cloud_timestep = time_step - t_now;
            }

            // Store old & accepted values in temporary variables
            y_old_v = yc_v;
            y_old_c = yc_c;

            // Reset error
            err_max_ratio = 0.0;

            // ---------------------------------------------------
            // k1
            dqdt(Pressure_d,
                 Cp_d,
                 y,
                 f,
                 p_vap,
                 tau_deep_d,
                 q_v_deep,
                 q_s,
                 cloud_timestep,
                 id,
                 icloud,
                 ntr_cloud,
                 lev,
                 nv,
                 deep_flag);

            // k1 for vapor/condensate
            k1_v = f[idx_v];
            k1_c = f[idx_c];

            // y_in = y + 0.5 * dt * k1
            y_in_v = y_old_v + 0.5 * cloud_timestep * k1_v;
            y_in_c = y_old_c + 0.5 * cloud_timestep * k1_c;

            y[idx_v] = y_in_v;
            y[idx_c] = y_in_c;

            // ---------------------------------------------------
            // k2
            dqdt(Pressure_d,
                 Cp_d,
                 y,
                 f,
                 p_vap,
                 tau_deep_d,
                 q_v_deep,
                 q_s,
                 cloud_timestep,
                 id,
                 icloud,
                 ntr_cloud,
                 lev,
                 nv,
                 deep_flag);

            k2_v = f[idx_v];
            k2_c = f[idx_c];

            // y_in = y + 0.75 * dt * k2
            y_in_v = y_old_v + 0.75 * cloud_timestep * k2_v;
            y_in_c = y_old_c + 0.75 * cloud_timestep * k2_c;

            // Overwrite y with y_in again
            y[idx_v] = y_in_v;
            y[idx_c] = y_in_c;

            // ---------------------------------------------------
            // k3
            dqdt(Pressure_d,
                 Cp_d,
                 y,
                 f,
                 p_vap,
                 tau_deep_d,
                 q_v_deep,
                 q_s,
                 cloud_timestep,
                 id,
                 icloud,
                 ntr_cloud,
                 lev,
                 nv,
                 deep_flag);

            k3_v = f[idx_v];
            k3_c = f[idx_c];

            // y_new = y + dt * (2/9*k1 + 1/3*k2 + 4/9*k3)
            y3_v = y_old_v + cloud_timestep * ((2.0 / 9.0) * k1_v + (1.0 / 3.0) * k2_v + (4.0 / 9.0) * k3_v);
            y3_c = y_old_c + cloud_timestep * ((2.0 / 9.0) * k1_c + (1.0 / 3.0) * k2_c + (4.0 / 9.0) * k3_c);
            
            // Embedded solution for error calculations
            // Keep in mind the actual embedded solution has a k4 term, this is an approximation
            y2_v = y_old_v + cloud_timestep * ((7.0 / 24.0) * k1_v + 0.25 * k2_v + (1.0 / 3.0) * k3_v);
            y2_c = y_old_c + cloud_timestep * ((7.0 / 24.0) * k1_c + 0.25 * k2_c + (1.0 / 3.0) * k3_c);
            
            // Calculate the error and keep the maximum error/tolerance
            tol_v = a_tol + r_tol * fabs(y3_v);
            tol_c = a_tol + r_tol * fabs(y3_c);
            err_max_ratio = fmax(fabs(y3_v-y2_v)/tol_v,
                                 fabs(y3_c-y2_c)/tol_c);
            
            // Accept / Reject the current time step
            accept = (err_max_ratio <= 1.0);
            // if (id == 0 ){
            //     printf("Iteration: %d | dt = %.1e s | E/T: %.1e \n", n_it, cloud_timestep, err_max_ratio);
            // }

            if (accept) { 
                // Commit new solutions
                yc_v   = y3_v;
                yc_c   = y3_c;
                y[idx_v] = yc_v;
                y[idx_c] = yc_c;

                // Update time
                t_now += cloud_timestep;
                n_it++;

                // Adjust time step size
                factor = pow(fmax(err_max_ratio,1.0e-16), -poww);
                cloud_timestep = safe * cloud_timestep * factor;
                cloud_timestep = fmin(cloud_timestep, time_step - t_now);
            } else {
                // Adjust time step size (shrink)
                factor = pow(err_max_ratio, -poww);
                cloud_timestep = safe * cloud_timestep * factor;
                // Check whether time step is getting too small
                if (cloud_timestep < dt_min) {
                    // Final fallback: take the step at dt_mn even if err>1
                    cloud_timestep = dt_min;
                    // commit anyway
                    yc_v = y3_v;                                 
                    yc_c = y3_c;
                    y[idx_v] = yc_v;
                    y[idx_c] = yc_c;
                    t_now += cloud_timestep;
                    n_it++;
                }else{
                    // Restore old values
                    yc_v = y_old_v;
                    yc_c = y_old_c;
                    y[idx_v] = yc_v;
                    y[idx_c] = yc_c;
                    // Retry time step with smaller dt
                    continue;   
                }
            }
            // Update solutions
            // y[idx_v] = yc_v;
            // y[idx_c] = yc_c;
            // // Update the local time
            // t_now += cloud_timestep;
            // n_it++;
        }

        // Ensure that tracers are non-negative
        // y[idx_v] = fmax(y[idx_v], 1e-99);
        // y[idx_c] = fmax(y[idx_c], 1e-99);

        // Final results, convert back from VMR to MMR        
        // Handle bottom boundary and clamp very small values
        if (lev == 0){
            q_c = 1e-30;
            q_v = fmin(q_v_deep * eps, q_s * eps);
        }else{
            q_c = fmax(y[idx_c] * eps, 1e-30);
            q_v = fmax(y[idx_v] * eps, 1e-30);
        }

        if (q_v < 0.0 || y[idx_v] < 0.0 || isnan(q_v)) {
            printf("Negative tracer concentrations generated!! \n");
            printf("q_v_vmr = %.3e  | q_v = %.3e | q_v_calc = %.3e | eps = %.3f \n",
                   q_v,
                   y[idx_v],
                   y[idx_v] * eps,
                   eps);
        }
        if (q_c < 0.0 || y[idx_c] < 0.0 || isnan(q_c)) {
            printf("Negative tracer concentrations generated!! \n");
            printf("q_c_vmr = %.3e  | q_c = %.3e | q_c_calc = %.3e | eps = %.3f \n",
                   q_c,
                   y[idx_c],
                   y[idx_c] * eps,
                   eps);
        }
        // // Change in atmospheric temperature from latent heat
        // dT = y[idx_T] - Temperature_d[id * nv + lev];

        // Store updated mixing ratios back to global memory
        tracer_cloud_d[idx_v] = q_v * Rho_d[id * nv + lev];
        tracer_cloud_d[idx_c] = q_c * Rho_d[id * nv + lev];
    }
}

__global__ void mini_cloud_settling_velocity(double *Pressure_d,    // Pressure (cell centers) [Pa]
                                             double *Temperature_d, // Temperature (cell centers)[K]
                                             double *Rho_d,         // Density [kg/m3]
                                             double *Cp_d,          // Specific heat capacity [J/kg/K]
                                             double *Rd_d,          // Gas constant [J/kg/K]
                                             double  Gravit,        // Gravity [m/s2]
                                             double *Altitude_d,    // Altitudes of the layers
                                             double *Altitudeh_d,   // Altitudes of the interfaces
                                             double *vf_d,          // Gravitational settling velocity [m/s]
                                             double *nd_atm_d,      // Background number density [m-3]
                                             double *rho_pd_d,      // Particle density [kg/m3]
                                             double  sigma_d,       // Particle size distribution std
                                             double  rm_d,          // Median particle size [m]
                                             double  time_step,     // time step [s]
                                             int     num)           // Number of columns
{
    // Get CUDA grid parameters
    int id  = blockIdx.x * blockDim.x + threadIdx.x;
    int nv  = gridDim.y;
    int lev = blockIdx.y;
    int n_cloud = gridDim.z;
    int icloud = blockIdx.z;

    // Constants
    const double R_UNIV_th = 8.31446261815324; // universal gas constant in [J/(K mol)]
    const double K_B_th    = 1.38064852e-23;   // Boltzmann constant in [J/K]
    const double AMU_th    = 1.6605390666e-27; // atomic mass unit in [kg]

    // Diameter [m], LJ potential and molecular weight [kg /mol] for background gases
    double d_H2 = 2.827e-8 / 100.0, LJ_H2 = 59.7 * K_B_th, molg_H2 = 2.01588 / 1000.0;

    // Temporary variables for computations
    double       mu, mu_dimless, eta, mfp, r_c, Kn, beta;
    const double r_seed = 1e-9; // [m]

    if (id < num) {

        // Molar mass [kg/mol]
        mu = R_UNIV_th / Rd_d[id * nv + lev];

        // Mean molecular weight [amu]
        mu_dimless = K_B_th / AMU_th / Rd_d[id * nv + lev];

        // Number density [m-3] of layer
        nd_atm_d[id * nv + lev] =
            Pressure_d[id * nv + lev] / (K_B_th * Temperature_d[id * nv + lev]);

        // Calculate dynamical viscosity for this layer (This expression needs updating!!)
        eta = (5.0 / 16.0)
              * (sqrt(M_PI * (mu_dimless * AMU_th) * K_B_th * Temperature_d[id * nv + lev])
                 / (M_PI * pow(d_H2, 2)))
              * (pow(((K_B_th * Temperature_d[id * nv + lev]) / LJ_H2), 0.16) / 1.22);

        // Calculate mean free path for this layer
        mfp = (2.0 * eta / Rho_d[id * nv + lev])
              * sqrt((M_PI * mu) / (8.0 * R_UNIV_th * Temperature_d[id * nv + lev]));

        // Volume (or mass) weighted mean radius of particle assuming log-normal distribution
        r_c = fmax(rm_d * exp(7.0 / 2.0 * pow(log(sigma_d), 2)), r_seed);

        // Knudsen number
        Kn = mfp / r_c;

        // Cunningham slip factor
        beta = 1.0 + Kn * (1.257 + 0.4 * exp(-1.1 / Kn));

        // Indexing variables
        const int STRIDE_ID_SP  = nv * n_cloud;
        const int STRIDE_LEV_SP = n_cloud;
        int idx_sp              = id * STRIDE_ID_SP + lev * STRIDE_LEV_SP + icloud;

        // Settling velocity
        vf_d[idx_sp] =
            (2.0 * beta * Gravit * pow(r_c, 2) * rho_pd_d[icloud]) / (9.0 * eta)
            * pow(1.0 + pow((0.45 * Gravit * pow(r_c, 3) * Rho_d[id * nv + lev] * rho_pd_d[icloud])/ (54.0 * pow(eta, 2)),0.4),-1.25);
        // if (id == 0 && lev == 10){
        //     printf("Vertical settling velocity for species %d v_f = %.2e m/s \n", icloud, vf_d[idx_sp]);
        // }
    }
}

__global__ void vert_adv_exp_maccormack(double *Rho_d,          // Density [kg/m3]
                                        double *Altitude_d,     // Altitudes of the layers
                                        double *Altitudeh_d,    // Altitudes of the interfaces
                                        double *tracer_cloud_d, // Cloud tracers
                                        double *vf_d,      // Gravitational settling velocity [m/s] - Cell-centers
                                        double *vfh_d,     // Gravitational settling velocity [m/s] - Interfaces
                                        double *nd_atm_d,  // Number density of each layer [m-3]
                                        double  Gravit,    // Gravity [m/s2]
                                        double  time_step, // Time step [s]
                                        int     ntr_cloud, // Number of cloud tracers
                                        int     n_cloud,   // Number of cloud species
                                        int     num,       // Number of columns
                                        int     nv)            // Number of vertical layers
{
    // Get column id
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    // Declare constants
    const double CFL = 0.9;

    // Working variables
    int       idx_ci, idx_cp, idx_cn;
    double    q_ci, q_cp, q_cn;
    double    q_c_pred, q_c_corr, q_c_pred_prev;
    double    dt, delz, courant, sig, r;
    const int STRIDE_ID     = nv * ntr_cloud;
    const int STRIDE_LEV    = ntr_cloud;
    const int STRIDE_ID_SP  = nv * n_cloud;
    const int STRIDE_ID_FSP = (nv + 1) * n_cloud;
    const int STRIDE_LEV_SP = n_cloud;
    int idx_fsp;

    if (id < num) {

        // Simple-average vertical settling velocity from layers to levels
        for (int icloud = 0; icloud < n_cloud; icloud++){
            
            // Top interface (lev = nv)
            idx_fsp = id * STRIDE_ID_FSP + nv * STRIDE_LEV_SP + icloud;
            vfh_d[idx_fsp] = vf_d[id * STRIDE_ID_SP + (nv - 1) * STRIDE_LEV_SP + icloud];

            for (int lev = nv - 1; lev >= 1; --lev) {

                idx_fsp = id * STRIDE_ID_FSP + lev * STRIDE_LEV_SP + icloud;

                int idx_sp   = id * STRIDE_ID_SP + lev * STRIDE_LEV_SP + icloud;
                int idx_spn  = idx_sp - STRIDE_LEV_SP;

                vfh_d[idx_fsp] = 0.5 * (vf_d[idx_sp] + vf_d[idx_spn]);
            }
            
            // Bottom interface (lev = 0)
            idx_fsp = id * STRIDE_ID_FSP + icloud;                    
            vfh_d[idx_fsp] = vf_d[id * STRIDE_ID_SP + icloud];
            
        }

        // Find minimum timestep that allows the CFL condition
        dt = time_step;
        for (int icloud = 0; icloud < n_cloud; ++icloud) {
            for (int lev = 0; lev < nv; lev++) {
                idx_fsp  = id * STRIDE_ID_FSP + lev * STRIDE_LEV_SP + icloud;
                delz    = Altitudeh_d[lev + 1] - Altitudeh_d[lev];
                dt      = fmin(dt, CFL * (delz / abs(vfh_d[idx_fsp])));
            }
        }

        // MacCormack integrator
        double t_now = 0.0;
        int    n_it  = 0;
        while ((t_now < time_step) && n_it < 10000) {

            // If next time step overshoots, adjust it
            if (t_now + dt >= time_step) {
                dt = time_step - t_now;
            }

            for (int icloud = 0; icloud < n_cloud; icloud++) {
                // Indexing variables
                int base_var = 2 * icloud;

                for (int lev = nv - 1; lev >= 0; lev--) {
                    
                    idx_fsp  = id * STRIDE_ID_FSP + lev * STRIDE_LEV_SP + icloud;
                    // Find the Courant number
                    delz    = Altitudeh_d[lev + 1] - Altitudeh_d[lev];
                    courant = abs(vfh_d[idx_fsp]) * dt / delz;

                    // Condensate index for the given cloud species
                    idx_ci = id * STRIDE_ID + lev * STRIDE_LEV + base_var + 1;
                    idx_cp = id * STRIDE_ID + (lev - 1) * STRIDE_LEV + base_var + 1;
                    idx_cn = id * STRIDE_ID + (lev + 1) * STRIDE_LEV + base_var + 1;

                    // Retrieve condensate mass-mixing ratios
                    // We are working with rho * q_c because that is what the dynamical core evolves.
                    q_ci = tracer_cloud_d[idx_ci] / Rho_d[id * nv + lev] * nd_atm_d[id * nv + lev];

                    // Read from lev+1 unless we're at the top (lev == nv-1)
                    if (lev == 0) {
                        // Bottom layer has no "below" neighbor, fallback to q_ci
                        q_cp = q_ci;
                    }
                    else {
                        // Valid neighbor above
                        q_cp = tracer_cloud_d[idx_cp] / Rho_d[id * nv + (lev - 1)]
                               * nd_atm_d[id * nv + lev - 1];
                    }

                    // Read from lev-1 unless we're at the bottom (lev == 0)
                    if (lev == nv - 1) {
                        // Top layer has no "above" neighbor, fallback to q_ci
                        q_cn = q_ci;
                    }
                    else {
                        // Valid neighbor below
                        q_cn = tracer_cloud_d[idx_cn] / Rho_d[id * nv + (lev + 1)]
                               * nd_atm_d[id * nv + lev + 1];
                    }

                    // Apply Koren slope limiter
                    if ((lev == 0) || (lev == nv - 1)) {
                        sig = 0.0;
                    }
                    else {
                        r   = (q_ci - q_cn) / (q_cp - q_ci + 1e-10);
                        sig = fmax(0.0, fmin(fmin(2.0 * r, (1.0 + 2.0 * r) / 3.0), 2.0));
                    }

                    // Perform MacCormack step //

                    // Predictor step
                    q_c_pred = q_ci - sig * courant * (q_cp - q_ci);

                    // Set the initial predictor value
                    if (lev == nv - 1) {
                        q_c_pred_prev = q_ci;
                    }

                    if (lev == 0) {
                        // Apply boundary conditions
                        q_c_corr = 1e-30;
                    }
                    else {
                        // Corrector step
                        q_c_corr = 0.5 * (q_ci + q_c_pred - courant * (q_c_pred - q_c_pred_prev));
                    }

                    // Ensure that tracers are non-negative
                    q_c_corr = fmax(q_c_corr, 1e-30);

                    // Update the previous predictor value
                    q_c_pred_prev = q_c_pred;

                    // Update the tracer array
                    tracer_cloud_d[idx_ci] =
                        q_c_corr * Rho_d[id * nv + lev] / nd_atm_d[id * nv + lev];
                }
            }

            // Update time
            t_now += dt;
            n_it++;
        }
        // Enforce boundary condition at bottom boundary (lev == 0)
        for (int icloud = 0; icloud < n_cloud; icloud++) {
            // Indexing variables
            int base_var = 2 * icloud;
            int lev = 0;
            idx_ci = id * STRIDE_ID + lev * STRIDE_LEV + base_var + 1;
            tracer_cloud_d[idx_ci] = 1e-30 * Rho_d[id * nv + lev];
        }
    }
}

__device__ inline void compute_fluxes_column(double *Altitude_d,  // Altitudes of the cell-centers
                                             double *Altitudeh_d, // Altitudes of the interfaces  
                                             double *Kzzh_d,      // Eddy diffusivity [m2 s-1]
                                             double *Rho_d,       // Density at cell-centers
                                             double *Rhoh_d,      // Density at interfaces
                                             double *q_in,        // Work arrays
                                             double *flux_out,
                                             int     id,         // Column id
                                             int     icloud,     
                                             int     ntr_cloud,  // Number of cloud tracers
                                             int     lev,        // Current vertical level
                                             int     nv)         // Number of vertical layers   
{
    // Indexing variables
    int idx_mid, idx_p, idx_n; 
    
    // Indexing constants
    const int STRIDE_ID  = nv * ntr_cloud;
    const int STRIDE_LEV = ntr_cloud;
    const int base_var   = 2 * icloud;

    // Work variables
    double delz, delz_mid, delz_mid_above;
    double phit, phil;

    // Vapour & condensate evolve in the same manner, so we just loop over them
    for (int itr = 0; itr < 2; ++itr) {

        idx_mid = id * STRIDE_ID + lev * STRIDE_LEV + base_var + itr;
        idx_p = id * STRIDE_ID + (lev + 1) * STRIDE_LEV + base_var + itr;
        idx_n = id * STRIDE_ID + (lev - 1) * STRIDE_LEV + base_var + itr;

        // Zero fluxes at the boundaries
        if (lev == 0 || lev == nv - 1){
            flux_out[idx_mid] = 0.0;
        }else{
            delz = Altitudeh_d[lev + 1] - Altitudeh_d[lev];
            delz_mid = Altitude_d[lev] - Altitude_d[lev - 1];
            delz_mid_above = Altitude_d[lev + 1] - Altitude_d[lev];
            
            phit = Rhoh_d[id * (nv+1) + lev - 1] * Kzzh_d[id * (nv+1) + lev - 1]  * (q_in[idx_n] - q_in[idx_mid]) / delz_mid;
            phil = Rhoh_d[id * (nv+1) + lev]   * Kzzh_d[id * (nv+1) + lev] * (q_in[idx_mid] - q_in[idx_p]) / delz_mid_above;

            flux_out[idx_mid] = (phit - phil) / delz / Rho_d[id * nv + lev];
        }
    }
}
__global__ void layers2interfaces(double *Rho_d,          // Density [kg/m3] - layers
                                  double *Rhoh_d,         // Density at levels
                                  double *Kzz_d,          // Eddy diffusivity [m2/s]
                                  double *Kzzh_d,
                                  int     num,            // Number of columns
                                  int     nv)
{
    // Get column id
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (id < num){
            // Simple-average Kzz and rho from layers to levels //
            // Top interface (lev = nv)
            Kzzh_d[id * (nv+1) + nv] =  Kzz_d[id * nv + nv -1];
            Rhoh_d[id * (nv+1) + nv] =  Rho_d[id * nv + nv -1];
            for (int lev = nv - 1; lev >= 1; lev--) {
                Kzzh_d[id * (nv+1) + lev] = 0.5 * (Kzz_d[id * nv + lev] + Kzz_d[id * nv + lev - 1]);
                Rhoh_d[id * (nv+1) + lev] = 0.5 * (Rho_d[id * nv + lev] + Rho_d[id * nv + lev - 1]);
            }
            // Bottom interface (lev = 0)
            Kzzh_d[id * (nv + 1) + 0] = Kzz_d[id * nv  + 0];
            Rhoh_d[id * (nv + 1) + 0] = Rho_d[id * nv  + 0];
    }
}
__global__ void vert_diff_exp_linear(double *Rho_d,          // Density [kg/m3] - layers
                              double *Rhoh_d,         // Density at levels
                              double *Altitude_d,     // Altitudes of the layers
                              double *Altitudeh_d,    // Altitudes of the interfaces
                              double *tracer_cloud_d, // Cloud tracers
                              double *Kzz_d,          // Eddy diffusivity [m2/s]
                              double *Kzzh_d, 
                              double *y,              // Work arrays
                              double *f,
                              double *y_old_d,
                              double *y_stage_d,
                              double *k1_d,
                              double *k2_d,
                              double  Gravit,         // Gravity [m/s2]
                              double  time_step,      // Time step [s]
                              int     num,            // Number of columns
                              int     nv)             // Number of vertical layers
{
    // Get column id
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    // Get cloud species
    int n_cloud = gridDim.z;
    int icloud = blockIdx.z;

    int ntr_cloud = 2 * n_cloud;

    // Working variables
    double q_v, q_c;
    double dt, dt_max, delz;
    
    // Local working variables for the Bogacki–Shampine (order 3) Runge-Kutta method
    double y3_v, y3_c, y2_v, y2_c;

    // Indexing constants
    int idx_v, idx_c, base_var;
    const int STRIDE_ID  = nv * ntr_cloud;
    const int STRIDE_LEV = ntr_cloud;

    // Error tolerances
    bool accept;
    double factor, err_max_ratio, tol_c, tol_v;
    const double CFL = 0.9;
    const double poww = 0.6;
    const double a_tol = 1e-20;
    const double safe = 0.9;
    const double r_tol = 5e-2;
    const double dt_min = 1e-1; 
    

    if (id < num){

        // Find minimum timestep that allows the CFL condition
        dt_max = time_step;
        for (int lev = 0; lev < nv; lev++) {
            delz    = Altitudeh_d[lev + 1] - Altitudeh_d[lev];
            dt_max      = fmin(dt_max, CFL * (delz * delz / (2.0 * Kzz_d[id * nv + lev])));
        }
        dt = dt_max;
        if (id == 0){
            printf("Linear access \n");
            printf("    dt = %.3e \n", dt);
        }
        // Read tracers from memory
        // Indexing variables
        base_var   = 2 * icloud;
        for (int lev = nv - 1; lev >= 0; lev--){
            
            // Assign vapor and condensate indices to variables for consistency
            idx_v = id * STRIDE_ID + lev * STRIDE_LEV + base_var + 0;
            idx_c = id * STRIDE_ID + lev * STRIDE_LEV + base_var + 1;
            
            // Retrieve vapor and condensate mass-mixing ratios 
            q_v = tracer_cloud_d[idx_v] / Rho_d[id * nv + lev];

            // Apply boundary conditions
            if (lev == 0){
                q_c = 1e-30;
            }else{    
                q_c = tracer_cloud_d[idx_c] / Rho_d[id * nv + lev];
            }

            // Ensure that tracers are non-negative
            q_v = fmax(q_v, 1e-30);
            q_c = fmax(q_c, 1e-30);

            // Initial conditions
            y[idx_v] = q_v;
            y[idx_c] = q_c;
        }

        // -----------------------------------------------------------------
        //  Adaptive Bogacki–Shampine loop
        // -----------------------------------------------------------------
        double t_now = 0.0;
        int    n_it  = 0;
        while ( t_now < time_step && n_it < 10000 ) {

            // Adjust time step so we don't overshoot
            if (t_now + dt > time_step){
                dt = time_step - t_now;
            } 

            // Backup first/accepted state into y_old
            // This is flattened indexing, faster when copying
            for (int lev = nv-1; lev >= 0; --lev) {
                idx_v = id * STRIDE_ID + lev * STRIDE_LEV + base_var + 0;
                idx_c = id * STRIDE_ID + lev * STRIDE_LEV + base_var + 1;
                y_old_d[idx_v] = y[idx_v];
                y_old_d[idx_c] = y[idx_c];
            }   
                
            // Reset error
            err_max_ratio = 0.0;

            // =============================================================
            // Stage-1  :  y_half = y + 0.5 * dt * k1
            // =============================================================
            for (int lev = nv-1; lev >= 0; --lev) {

                idx_v = id*STRIDE_ID + lev*STRIDE_LEV + base_var;
                idx_c = idx_v + 1;

                // k1
                compute_fluxes_column(Altitude_d,
                                        Altitudeh_d,
                                        Kzzh_d,
                                        Rho_d,
                                        Rhoh_d,
                                        y,
                                        f,
                                        id,
                                        icloud,
                                        ntr_cloud,
                                        lev,
                                        nv);

                k1_d[idx_v] = f[idx_v];
                k1_d[idx_c] = f[idx_c];

                // y_stage_d = y + 0.5 * dt * k1
                y_stage_d[idx_v] = y[idx_v] + 0.5 * dt * k1_d[idx_v];
                y_stage_d[idx_c] = y[idx_c] + 0.5 * dt * k1_d[idx_c];
            }
            // Store this stage's solutions in y so you reuse y_stage
            for (int lev = nv-1; lev >= 0; --lev) {
                idx_v = id * STRIDE_ID + lev * STRIDE_LEV + base_var + 0;
                idx_c = id * STRIDE_ID + lev * STRIDE_LEV + base_var + 1;
                y[idx_v] = y_stage_d[idx_v];
                y[idx_c] = y_stage_d[idx_c];
            }   
                
            // =============================================================
            // Stage-2  :  y_3q = y_old + 0.75 * dt * k2
            // =============================================================
            for (int lev = nv-1; lev >= 0; --lev) {

                idx_v = id*STRIDE_ID + lev*STRIDE_LEV + base_var;
                idx_c = idx_v + 1;

                // k2
                compute_fluxes_column(Altitude_d,
                                        Altitudeh_d,
                                        Kzzh_d,
                                        Rho_d,
                                        Rhoh_d,
                                        y,
                                        f,
                                        id,
                                        icloud,
                                        ntr_cloud,
                                        lev,
                                        nv);

                k2_d[idx_v] = f[idx_v];
                k2_d[idx_c] = f[idx_c];

                // y_stage_d = y_old + 0.75 * dt * k2
                y_stage_d[idx_v] = y_old_d[idx_v] + 0.75 * dt * k2_d[idx_v];
                y_stage_d[idx_c] = y_old_d[idx_c] + 0.75 * dt * k2_d[idx_c];
            }
            
            // Store this stage's solutions in y so you reuse y_stage
            for (int lev = nv-1; lev >= 0; --lev) {
                idx_v = id * STRIDE_ID + lev * STRIDE_LEV + base_var + 0;
                idx_c = id * STRIDE_ID + lev * STRIDE_LEV + base_var + 1;
                y[idx_v] = y_stage_d[idx_v];
                y[idx_c] = y_stage_d[idx_c];
            }   
                
            // =============================================================
            // Stage-3  :  full step + embedded error
            // =============================================================
            for (int lev = nv-1; lev >= 0; --lev) {

                idx_v = id*STRIDE_ID + lev*STRIDE_LEV + base_var;
                idx_c = idx_v + 1;

                // k3
                compute_fluxes_column(Altitude_d,
                                        Altitudeh_d,
                                        Kzzh_d,
                                        Rho_d,
                                        Rhoh_d,
                                        y,
                                        f,
                                        id,
                                        icloud,
                                        ntr_cloud,
                                        lev,
                                        nv);

                // y_new = y_old + dt * (2/9 * k1 + 1/3 * k2 + 4/9 * k3)
                y3_v = y_old_d[idx_v]
                    + dt * ((2.0 / 9.0) * k1_d[idx_v] + (1.0 / 3.0) * k2_d[idx_v] + (4.0 / 9.0) * f[idx_v]);

                y3_c = y_old_d[idx_c]
                    + dt * ((2.0 / 9.0) * k1_d[idx_c] + (1.0 / 3.0) * k2_d[idx_c] + (4.0 / 9.0) * f[idx_c]);
                
                // y_embedded = y_old + dt * (7/24 * k1 + 1/4 * k2 + 1/3 * k3) 
                // Keep in mind the actual embedded solution has a k4 term, this is an approximation
                y2_v = y_old_d[idx_v]
                    + dt * ((7.0 / 24.0) * k1_d[idx_v] + 0.25 * k2_d[idx_v] + (1.0 / 3.0) * f[idx_v]);

                y2_c = y_old_d[idx_c]
                    + dt * ((7.0 / 24.0) * k1_d[idx_c] + 0.25 * k2_d[idx_c] + (1.0 / 3.0) * f[idx_c]);

                // Store candidate solution in y_stage
                y_stage_d[idx_v] = y3_v;
                y_stage_d[idx_c] = y3_c;

                // Calculate the error and keep the maximum error/tolerance
                tol_v = a_tol + r_tol * fabs(y3_v);
                tol_c = a_tol + r_tol * fabs(y3_c);
                err_max_ratio = fmax(err_max_ratio,
                                    fmax(fabs(y3_v - y2_v) / tol_v,
                                            fabs(y3_c - y2_c) / tol_c));
            }

            // ============================================================
            // Accept / reject this time step
            // ============================================================
            accept = (err_max_ratio <= 1.0);
            // if (id == 0 && icloud == 0){
            //     printf("Iteration: %d | dt = %.1e s | E/T: %.1e \n", n_it, dt, err_max_ratio);
            // }
            
            if (accept) {
                // Commit y_stage to the actual solutions array y
                for (int lev = nv-1; lev >= 0; --lev) {
                    idx_v = id * STRIDE_ID + lev * STRIDE_LEV + base_var + 0;
                    idx_c = id * STRIDE_ID + lev * STRIDE_LEV + base_var + 1;
                    y[idx_v] = y_stage_d[idx_v];
                    y[idx_c] = y_stage_d[idx_c];
                }   
                // Update time
                t_now += dt;
                n_it++;

                // Adjust time step
                factor = pow(fmax(err_max_ratio,1.0e-16), -poww);
                dt     = safe * dt * factor;
                dt     = fmin(dt, dt_max);

            } else {
                // Adjust time step (shrink)
                factor = pow(err_max_ratio, -poww);
                dt     = safe * dt * factor;
                
                // Check whether time step is getting too small
                if (dt < dt_min) {
                    dt = dt_min;
                    // Commit the candidate solution even though err>1
                    for (int lev = nv-1; lev >= 0; --lev) {
                        idx_v = id * STRIDE_ID + lev * STRIDE_LEV + base_var + 0;
                        idx_c = id * STRIDE_ID + lev * STRIDE_LEV + base_var + 1;
                        y[idx_v] = y_stage_d[idx_v];
                        y[idx_c] = y_stage_d[idx_c];
                    }   
                    t_now += dt;
                    n_it++;                         
                }else{
                    // Overwrite y with y_old, restoring old solutions
                    for (int lev = nv-1; lev >= 0; --lev) {
                        idx_v = id * STRIDE_ID + lev * STRIDE_LEV + base_var + 0;
                        idx_c = id * STRIDE_ID + lev * STRIDE_LEV + base_var + 1;
                        y[idx_v] = y_old_d[idx_v];
                        y[idx_c] = y_old_d[idx_c];
                    }   
                    // Retry with a smaller timestep
                    continue;
                }

            }
        }


        // Update the tracer arrays with final solutions
        for (int lev = nv - 1; lev >= 0; lev--){
            // Assign vapor and condensate indices to variables for consistency
            idx_v = id * STRIDE_ID + lev * STRIDE_LEV + base_var + 0;
            idx_c = id * STRIDE_ID + lev * STRIDE_LEV + base_var + 1;
            
            // Handle boundary conditions
            if (lev == 0){
                q_c = 1e-30;
            }else{
                q_c = y[idx_c];
            }
            q_v = y[idx_v];

            // Checking if everything is stable
            if (q_v < 0.0 || y[idx_v] < 0.0 || isnan(q_v)) {
                printf("Negative tracer concentrations generated!! \n");
            }
            if (q_c < 0.0 || y[idx_c] < 0.0 || isnan(q_c)) {
                printf("Negative tracer concentrations generated!! \n");
            }

            // Store updated mixing ratios back to global memory
            tracer_cloud_d[idx_v] = q_v * Rho_d[id * nv + lev];
            tracer_cloud_d[idx_c] = q_c * Rho_d[id * nv + lev];
        }
    } 
}

__device__ inline double compute_fluxes_level(const double* __restrict__ q,     // one tracer, shared
                                            const double* __restrict__ rho,   // shared
                                            const double* __restrict__ kzz,   // shared
                                            const double* __restrict__ dz,    // shared
                                            int lev,
                                            int nv)
{
    // Handle boundary conditions
    if (lev == 0 || lev == nv-1){
        return 0.0;          
    } 

    // Declare working variables
    double kzzh_up, kzzh_dn, rhoh_up, rhoh_dn;
    double dz_mid_up, dz_mid_dn; 
    double grad_up, grad_dn;
    double flux_out; 

    // Find interface values on the fly
    kzzh_dn = 0.5*(kzz[lev] + kzz[lev-1]);
    kzzh_up = 0.5*(kzz[lev] + kzz[lev+1]);
    rhoh_dn = 0.5*(rho[lev] + rho[lev-1]);
    rhoh_up = 0.5*(rho[lev] + rho[lev+1]);

    // Differences of altitude centers
    dz_mid_dn = 0.5*(dz[lev] + dz[lev-1]);
    dz_mid_up = 0.5*(dz[lev] + dz[lev+1]);

    grad_dn = (q[lev-1] - q[lev  ]) / dz_mid_dn;
    grad_up = (q[lev  ] - q[lev+1]) / dz_mid_up;

    flux_out = (rhoh_dn * kzzh_dn * grad_dn - rhoh_up * kzzh_up * grad_up) / (rho[lev] * dz[lev]);

    return flux_out;
}

__global__ void vert_diff_exp_parallel(double *Rho_d,          // Density [kg/m3] - layers
                              double *Altitude_d,     // Altitudes of the layers
                              double *Altitudeh_d,    // Altitudes of the interfaces
                              double *tracer_cloud_d, // Cloud tracers
                              double *Kzz_d,          // Eddy diffusivity [m2/s]
                              double  Gravit,         // Gravity [m/s2]
                              double  time_step,      // Time step [s]
                              int     num,            // Number of columns
                              int     nv)             // Number of vertical layers
{

    // Get cloud species
    int n_cloud = gridDim.z;
    int icloud = blockIdx.z;

    int ntr_cloud = 2 * n_cloud;

    // Get column id and the threads assigned to it
    const int id  = blockIdx.x;     // horizontal column id
    const int lev  = threadIdx.x;   // vertical level handled by this thread
    const int threads_per_block = blockDim.x;

    // Dynamic shared array that all the threads in this block have access to
    extern __shared__ double sh[];  
    double* qv  = &sh[0 * threads_per_block]; 
    double* qc  = &sh[1 * threads_per_block];
    double* rho = &sh[2 * threads_per_block];
    double* kzz = &sh[3 * threads_per_block];
    double* dz  = &sh[4 * threads_per_block];
    double* dt_sh  = &sh[5 * threads_per_block];

     // Indexing variables
    int idx_v, idx_c, base_var;
    const int STRIDE_ID  = nv * ntr_cloud;
    const int STRIDE_LEV = ntr_cloud;

    // Error tolerances
    bool accept;
    double factor, err_max_ratio, tol_c, tol_v;
    const double CFL = 0.9;
    const double poww = 0.6;
    const double a_tol = 1e-20;
    const double safe = 0.9;
    const double r_tol = 5e-2;
    const double dt_min = 1e-1; 

    // Local variables
    double dt, dt_max; 
    double k1_c, k1_v, k2_c, k2_v, k3_c, k3_v;
    double qv_s, qc_s, qv_old, qc_old;
    double qv_3, qc_3, qv_2, qc_2;

    // We launch more threads than nv (usually) padded to the next multiple of 32
    // so this is a thread safety measure.
    if (lev < nv){   
        // Assign vapor and condensate indices to variables for consistency
        base_var = 2 * icloud;
        idx_v    = id * STRIDE_ID + lev * STRIDE_LEV + base_var + 0;
        idx_c    = id * STRIDE_ID + lev * STRIDE_LEV + base_var + 1;   

        // Load global arrays to local shared arrays for a single column
        rho[lev] = Rho_d[id * nv + lev];
        kzz[lev] = Kzz_d[id * nv + lev];

        // Read MMRs and make sure they are non-negative
        qv[lev] = fmax(tracer_cloud_d[idx_v] / rho[lev], 1e-30);
        if (lev == 0){ // Impose boundary condition after dynamical core loop
            qc[lev] = 1e-30;
        }else{
            qc[lev] = fmax(tracer_cloud_d[idx_c] / rho[lev], 1e-30);
        }
        
        dz[lev]   = Altitudeh_d[lev+1] - Altitudeh_d[lev];
    }
    
    // Sync needs to be outside to hit all threads
    __syncthreads();             
    
    dt_max = time_step;
    // Find minimum timestep that allows the CFL condition
    if (lev < nv){
        dt_sh[lev] = fmin(dt_max, CFL * (dz[lev] * dz[lev] / (2.0 * kzz[lev])));
    }else{
        // Ensures that if a warp is larger than nv
        // we don't compare against garbage
        dt_sh[lev] = 1e30; 
    }
     
    __syncthreads();
   
    // Traverse the tree in a binary search style to find min()
    // Bitwise operation essentially divides i by 2
    for (int i = blockDim.x / 2; i > 0; i >>= 1){
        // Make sure we are comparing one half to the other
        if (lev < i){
            dt_sh[lev] = fmin(dt_sh[lev], dt_sh[lev + i]);
        }
        __syncthreads();
    }
    // Minimum value is now stored in index 0
    dt = dt_sh[0];

    // Adaptive Bogacki - Shampine loop
    double t_now = 0.0;
    int n_it = 0;
    while (t_now < time_step && n_it < 10000){

        // Adjust time step so we don't overshoot
        if (t_now + dt > time_step){
            dt = time_step - t_now;
        } 

        // Reset error
        err_max_ratio = 0.0;

        __syncthreads();

        if (lev < nv){
            // Store old & accepted values in temporary variables
            qv_old = qv[lev];
            qc_old = qc[lev];

            // =============================================================
            // Stage-1  :  y_half = y + 0.5 * dt * k1
            // =============================================================
            k1_v = compute_fluxes_level(qv, rho, kzz, dz, lev, nv);
            k1_c = compute_fluxes_level(qc, rho, kzz, dz, lev, nv);

            qv_s = qv_old + 0.5 * dt * k1_v;
            qc_s = qc_old + 0.5 * dt * k1_c;
        }

        __syncthreads();
        
        if (lev < nv){
            // Update arrays with this stage's values
            qv[lev] = qv_s;
            qc[lev] = qc_s;
        }

        __syncthreads();
        
        // =============================================================
        // Stage-2  :  y_3q = y_old + 0.75 * dt * k2
        // =============================================================

        if (lev < nv){
            k2_v = compute_fluxes_level(qv, rho, kzz, dz, lev, nv);
            k2_c = compute_fluxes_level(qc, rho, kzz, dz, lev, nv);

            // y_in = y + 0.75 * dt * k2
            qv_s = qv_old + 0.75 * dt * k2_v;
            qc_s = qc_old + 0.75 * dt * k2_c;
        }

        __syncthreads();
        
        if (lev < nv){
            // Update arrays with this stage's values
            qv[lev] = qv_s;
            qc[lev] = qc_s;
        }

        __syncthreads();

        // ==================================================================
        // Stage-3  :  y_full = y_old + dt * (2/9 * k1 + 1/3 * k2 + 4/9 * k3)
        // ==================================================================
        
        if (lev < nv){
            k3_v = compute_fluxes_level(qv, rho, kzz, dz, lev, nv);
            k3_c = compute_fluxes_level(qc, rho, kzz, dz, lev, nv);

            // y_new = y + dt * (2/9*k1 + 1/3*k2 + 4/9*k3)
            qv_3 = qv_old + dt * ((2.0 / 9.0) * k1_v + (1.0 / 3.0) * k2_v + (4.0 / 9.0) * k3_v);
            qc_3 = qc_old + dt * ((2.0 / 9.0) * k1_c + (1.0 / 3.0) * k2_c + (4.0 / 9.0) * k3_c);
            
            // Embedded solution for error calculations
            qv_2 = qv_old + dt * ((7.0 / 24.0) * k1_v + 0.25 * k2_v + (1.0 / 3.0) * k3_v);
            qc_2 = qc_old + dt * ((7.0 / 24.0) * k1_c + 0.25 * k2_c + (1.0 / 3.0) * k3_c);
        
        }
        
        __syncthreads();

        // Calculate and reduce the max error
        if (lev < nv){
            // Calculate the error and keep the maximum error/tolerance
            tol_v = a_tol + r_tol * fabs(qv_3);
            tol_c = a_tol + r_tol * fabs(qc_3);
            err_max_ratio = fmax(fabs(qv_3-qv_2)/tol_v,
                                 fabs(qc_3-qc_2)/tol_c);
            // if (t_now == 0.0){
            //     printf("Level : %d | Error: %.2e \n", lev, err_max_ratio);
            // }
            
            // Re-use the scratch array                            
            dt_sh[lev] = err_max_ratio;     
        }else{
            // Ensures that if a warp is larger than nv
            // we don't compare against garbage
            dt_sh[lev] = 1e-99; 
        }
        
        __syncthreads();
    
        // Traverse the tree in a binary search style to find min()
        // Bitwise operation essentially divides i by 2
        for (int i = blockDim.x / 2; i > 0; i >>= 1){
            // Make sure we are comparing one half to the other
            if (lev < i){
                dt_sh[lev] = fmax(dt_sh[lev], dt_sh[lev + i]);
            }
            __syncthreads();
        }
        // Maximum value is now stored in index 0
        err_max_ratio = dt_sh[0];

        __syncthreads();

        // Update the time step with a single thread and broadcast to others
        if (lev == 0) {       
            accept = (err_max_ratio <= 1.0);
            double new_dt;

            // Accept / Reject the current time step
            if (accept) {
                factor = pow(fmax(err_max_ratio, 1.0e-16), -poww);
                new_dt = fmin(safe * dt * factor, dt_max);
            }
            else {
                factor = pow(err_max_ratio, -poww);
                new_dt = safe * dt * factor;

                // If timestep is too small
                if (new_dt <= dt_min) {
                    accept  = true;
                    new_dt = dt_min;
                    err_max_ratio = 0.0;
                }
            }
            
            dt_sh[0] = accept;
            dt_sh[1] = new_dt;
        }
        __syncthreads();  

        // Broadcast column results to every thread
        accept = dt_sh[0];
        dt     = dt_sh[1];
        
        // if (id == 0 && lev == 0 && icloud == 0){
        //     printf("Iteration: %d | dt = %.1e s | E/T: %.1e \n", n_it, dt, err_max_ratio);
        // }
        
        
        if (accept) { 
            
            if (lev < nv){
                // Commit new solutions
                qv[lev] = qv_3;
                qc[lev] = qc_3;
            }

            // Update time
            t_now += dt;
            n_it++;
        } else {
            
            if (lev < nv){
                // Restore old values
                qv[lev] = qv_old;
                qc[lev] = qc_old;
            }
        }
        __syncthreads();
    }

    
    // Update the tracer arrays with final solutions
    if (lev < nv){   
        // Assign vapor and condensate indices to variables for consistency
        base_var = 2 * icloud;
        idx_v    = id * STRIDE_ID + lev * STRIDE_LEV + base_var + 0;
        idx_c    = id * STRIDE_ID + lev * STRIDE_LEV + base_var + 1;   
        
        // Handle boundary conditions
        if (lev == 0){
            qc[lev] = 1e-30;
        }

        // Store updated mixing ratios back to global memory
        tracer_cloud_d[idx_v] = qv[lev] * Rho_d[id * nv + lev];
        tracer_cloud_d[idx_c] = qc[lev] * Rho_d[id * nv + lev];
    }
}

__global__ void cloud_number_density(double *Rho_d,          // Density [kg/m3]
                                     double *Rd_d,           // Gas constant [J/kg/K]
                                     double *tracer_cloud_d, // Cloud tracers
                                     double *mol_w_sp_d,     // Molecular weight of species [kg/mol]
                                     double *n_tot_d,        // Cloud number density [m-3]
                                     double *rho_pd_d,       // Particle bulk density [kg/m3]
                                     double  sigma_d,        // Particle size distribution std
                                     double  rm_d,           // Median particle size [m]
                                     int     ntr_cloud,      // Number of cloud tracers
                                     int     num)            // Number of columns
{
    // Get CUDA grid parameters //
    // Column id
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    // Vertical layers
    int nv  = gridDim.y;
    int lev = blockIdx.y;
    // Cloud species
    int n_cloud = gridDim.z;
    int icloud  = blockIdx.z;

    // Universal gas constant
    const double R_UNIV_th = 8.31446261815324; //  [J*K-1*mol-1]

    // Declare working variables
    double q_c, eps, mu;

    // Indexing variables
    const int STRIDE_ID     = nv * ntr_cloud;
    const int STRIDE_LEV    = ntr_cloud;
    const int STRIDE_ID_SP  = nv * n_cloud;
    const int STRIDE_LEV_SP = n_cloud;

    if (id < num) {

        // Condensate tracer index
        int base_var = 2 * icloud;
        int idx_c    = id * STRIDE_ID + lev * STRIDE_LEV + base_var + 1;
        int idx_sp   = id * STRIDE_ID_SP + lev * STRIDE_LEV_SP + icloud;

        // Molar mass [kg/mol]
        mu = R_UNIV_th / Rd_d[id * nv + lev];

        // Conversion factor between MMR and VMR for cloud species. MMR --> VMR
        // The molecular weight fraction between the condensate and background gas
        eps = mol_w_sp_d[icloud]
              / mu; // Technically it's not even necessary to do this because we multiply it back...
        
        // Read-in the cloud condensate MMRs
        q_c = tracer_cloud_d[idx_c] / Rho_d[id * nv + lev] / eps;
        
        // Calculate total number density
        n_tot_d[idx_sp] = ((3.0 * q_c * eps * Rho_d[id * nv + lev])
                                  / (4.0 * M_PI * rho_pd_d[icloud] * rm_d * rm_d * rm_d))
                                 * exp((-9.0 / 2.0) * log(sigma_d) * log(sigma_d));
    }
}