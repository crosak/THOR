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
// ESP -  Exoclimes Simulation Platform. (version 1.0)
//
//
//
// Method: Clouds physics module
//
//
// Known limitations: - Runs in a single GPU.
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
//
//
////////////////////////////////////////////////////////////////////////

#pragma once

#include "phy_module_base.h"

class clouds : public phy_module_base
{
public:
    clouds();
    ~clouds();

    bool initialise_memory(const ESP &esp, device_RK_array_manager &phy_modules_core_arrays);

    bool initial_conditions(const ESP &esp, const SimulationSetup &sim, storage *s);

    bool dyn_core_loop_init(const ESP &esp);

    bool dyn_core_loop_slow_modes(const ESP             &esp,
                                  const SimulationSetup &sim,
                                  int                    nstep, // Step number
                                  double                 time_step);            // Time-step [s]

    bool dyn_core_loop_fast_modes(const ESP             &esp,
                                  const SimulationSetup &sim,
                                  int                    nstep, // Step number
                                  double                 times);                // Time-step [s]

    bool dyn_core_loop_end(const ESP &esp);

    bool phy_loop(ESP                   &esp,
                  const SimulationSetup &sim,
                  kernel_diagnostics    &diag,
                  int                    nstep, // Step number
                  double                 time_step);            // Time-step [s]

    bool store(const ESP &esp, storage &s);

    bool store_init(storage &s);

    bool configure(config_file &config_reader);

    virtual bool free_memory();

    void print_config();

private:
    // Number of cloud species
    int n_cloud   = 1;           // Number of cloud species
    int ntr_cloud = 2 * n_cloud; // Number of tracers (Vapour/Condensate)

    // host arrays
    double *tracer_cloud_h;

    // Device work arrays
    double *tracer_cloud_d;
    double *tracers_cloud_d;
    double *tracerk_cloud_d;
    double *difftr_cloud_d;

    double *vf_d;
    double *vfh_d;
    double *Kzzh_d;
    double *Rhoh_d;

    double *y_old_d;
    double *y_stage_d;
    double *k1_d;
    double *k2_d;
    double *y_d;
    double *f_d;

    // Cloud physics parameters
    double  tau_deep_d;
    double  tau_chem_d;
    double  sigma_d;
    double  rm_d;
    double *nd_atm_d;

    // Host-side vectors for reading in lists
    std::vector<std::string> sp;       // Cloud species identifiers
    std::vector<double>      mol_w_sp; // Molecular weights
    std::vector<double>      q_v_deep; // Deep VMR of individual cloud species
    std::vector<double>      rho_pd;   // Particle density of cloud species

    std::vector<int>         active_species;

    // Device-side arrays for the vector quantities
    int    *active_species_d; // Integer IDs for active species
    double *mol_w_sp_d;       // Molecular weights
    double *q_v_deep_d;       // Deep cloud VMRs
    double *rho_pd_d;         // Particle densities
};
