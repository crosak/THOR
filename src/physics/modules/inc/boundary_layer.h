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
// Method: Radiative transfer physics module
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
// Current Code Owner: Joao Mendonca, EEG. joao.mendonca@csh.unibe.ch
//
// History:
// Version Date       Comment
// ======= ====       =======
//
// 1.0     16/08/2017 Released version  (JM)
//
////////////////////////////////////////////////////////////////////////

#pragma once

#include "phy_module_base.h"

#define bl_type_default "RayleighHS"

#define LARGERiB 1e8

enum boundary_layer_types { RAYLEIGHHS = 0, MONINOBUKHOV = 1, EKMANSPIRAL = 2 };


class boundary_layer : public phy_module_base
{
public:
    boundary_layer();
    ~boundary_layer();

    bool initialise_memory(const ESP &esp, device_RK_array_manager &phy_modules_core_arrays);
    bool initial_conditions(const ESP &esp, const SimulationSetup &sim, storage *s);

    bool phy_loop(ESP &                  esp,
                  const SimulationSetup &sim,
                  int                    nstep, // Step number
                  double                 time_step);            // Time-step [s]

    bool store(const ESP &esp, storage &s);

    bool store_init(storage &s);

    bool configure(config_file &config_reader);

    virtual bool free_memory();

    void print_config();

private:
    // Config options
    boundary_layer_types bl_type;
    string               bl_type_str;

    double surf_drag_config = 1.0 / 86400.0; // surface drag coefficient
    double bl_sigma_config  = 0.7;           // sigma coord of boundary layer (% of surf pressure)

    double surf_drag;
    double bl_sigma;
    // double *dvdz_tmp;
    double *d2vdz2_tmp;
    double *atmp, *btmp, *ctmp, *cpr_tmp, *dtmp, *dpr_tmp;
    double  zbl;        // altitude of transition from BL to free atmosph
    int     bl_top_lev; // index of highest level inside BL
    double *RiB_d;      // bulk Richardson number

    double *KM_d; // momentum diffusivity (turbulence)
    double *KH_d; // heat diffusivity (turbulence)

    void BLSetup(const ESP &            esp,
                 const SimulationSetup &sim,
                 int                    bl_type_,
                 double                 surf_drag_,
                 double                 bl_sigma_);
};

__global__ void rayleighHS(double *Mh_d,
                           double *pressure_d,
                           double *Rho_d,
                           double *Altitude_d,
                           double  surf_drag,
                           double  bl_sigma,
                           double  Gravit,
                           double  time_step,
                           int     num);

__global__ void ConstKMEkman(double *Mh_d,
                             double *pressure_d,
                             double *Rho_d,
                             double *Altitude_d,
                             double *Altitudeh_d,
                             double *d2vdz2_tmp,
                             double  KMconst,
                             double  zbl,
                             double  time_step,
                             int     num,
                             int     nv);

__global__ void ConstKMEkman_Impl(double *Mh_d,
                                  double *pressure_d,
                                  double *Rho_d,
                                  double *Altitude_d,
                                  double *Altitudeh_d,
                                  double *atmp,
                                  double *btmp,
                                  double *ctmp,
                                  double *cpr_tmp,
                                  double *dtmp,
                                  double *dpr_tmp,
                                  double  KMconst,
                                  double  zbl,
                                  double  time_step,
                                  int     num,
                                  int     nv,
                                  int     bl_top_lev);

__global__ void CalcRiB(double *pressure_d,
                        double *Rho_d,
                        double *Mh_d,
                        double *Tsurface_d,
                        double *Altitude_d,
                        double *Altitudeh_d,
                        double  Rd,
                        double  Cp,
                        double  P_Ref,
                        double  Gravit,
                        double *RiB_d,
                        int     num,
                        int     nv);
