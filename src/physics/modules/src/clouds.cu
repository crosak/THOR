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
//
//
// Method: Cloud physics module
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
// 1.0     17/12/2024   (CA)
//
////////////////////////////////////////////////////////////////////////
#include "clouds.h"
#include "clouds_device.h"  // Simple clouds.
#include "thor_chemistry.h" // Stealing the dynamical core hijack functions

#include "binary_test.h"
#include "debug.h"
#include "directories.h"

#include <string>
#include <unordered_map>
#include <vector>

clouds::clouds() {
}

clouds::~clouds() {
}

void clouds::print_config() {
    log::printf("  Cloud module\n");

    // log::printf("    cloud_type                     = %s \n", cloud_type_str.c_str());
    log::printf("    Number of included cloud species  = %d.\n", n_cloud);
    log::printf("    Included cloud species:\n");
    for (const auto &species : sp) {
        log::printf("        %s\n", species.c_str());
    }
    log::printf("    Particle density [kg m-3]:\n");
    for (const auto &rho: rho_pd){
        log::printf("        %.3e\n", rho);
    }
    log::printf("    Deep replenishment timescale [s]  = %f.\n", tau_deep_d);
    log::printf("    Condensation timescale [s]        = %f.\n", tau_chem_d);
    log::printf("    Particle size distribution std.   = %f.\n", sigma_d);
    log::printf("    Median particle size [m]          = %.1e.\n", rm_d);
}

bool clouds::initialise_memory(const ESP &esp, device_RK_array_manager &phy_modules_core_arrays) {
    // Update the number of tracers with configuration file value (Is this really necessary?)
    ntr_cloud = 2 * n_cloud;
    printf("Number of tracers in the cloud module: %d \n", ntr_cloud);
    // Allocate host memory for tracer arrays
    tracer_cloud_h = (double *)malloc(esp.nv * esp.point_num * ntr_cloud * sizeof(double));

    // Allocate device memory for tracer arrays
    cudaMalloc((void **)&tracer_cloud_d, esp.nv * esp.point_num * ntr_cloud * sizeof(double));
    cudaMalloc((void **)&tracers_cloud_d, esp.nv * esp.point_num * ntr_cloud * sizeof(double));
    cudaMalloc((void **)&tracerk_cloud_d, esp.nv * esp.point_num * ntr_cloud * sizeof(double));
    cudaMalloc((void **)&difftr_cloud_d, esp.nv * esp.point_num * ntr_cloud * sizeof(double));

    // Cloud module physics arrays
    cudaMalloc((void **)&Kzzh_d, esp.nv * esp.point_num * sizeof(double));
    cudaMalloc((void **)&Rhoh_d, esp.nv * esp.point_num * sizeof(double));
    cudaMalloc((void **)&vf_d, esp.nv * esp.point_num * n_cloud * sizeof(double));
    cudaMalloc((void **)&vfh_d, esp.nvi * esp.point_num * n_cloud * sizeof(double));

    // Cloud module working arrays
    cudaMalloc((void **)&y_old_d, esp.nv * esp.point_num * ntr_cloud * sizeof(double));
    cudaMalloc((void **)&y_stage_d, esp.nv * esp.point_num * ntr_cloud * sizeof(double));
    cudaMalloc((void **)&y_d, esp.nv * esp.point_num * ntr_cloud * sizeof(double));
    cudaMalloc((void **)&f_d, esp.nv * esp.point_num * ntr_cloud * sizeof(double));
    cudaMalloc((void **)&k1_d, esp.nv * esp.point_num * ntr_cloud * sizeof(double));
    cudaMalloc((void **)&k2_d, esp.nv * esp.point_num * ntr_cloud * sizeof(double));
    cudaMalloc((void **)&nd_atm_d, esp.nv * esp.point_num * sizeof(double));

    // Allocate device memory for cloud properties
    cudaMalloc((void **)&q_v_deep_d, n_cloud * sizeof(double));
    cudaMemcpy(q_v_deep_d, q_v_deep.data(), n_cloud * sizeof(double), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&rho_pd_d, n_cloud * sizeof(double));
    cudaMemcpy(rho_pd_d, rho_pd.data(), n_cloud * sizeof(double), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&mol_w_sp_d, n_cloud * sizeof(double));
    cudaMemcpy(mol_w_sp_d, mol_w_sp.data(), n_cloud * sizeof(double), cudaMemcpyHostToDevice);

    // Print the contents of the `sp` vector to verify input
    std::cout << "Species vector from config (sp): ";
    for (const auto &name : sp) {
        std::cout << name << " ";
    }
    std::cout << std::endl;

    // Map to assign a unique integer to each species name
    std::unordered_map<std::string, int> species_map = {
        {"C", 0},      {"TiO2", 1},  {"Al2O3", 2},  {"Fe", 3},   {"Mg2SiO4", 4}, {"MgSiO3", 5},
        {"SiO2", 6},   {"SiO", 7},   {"Cr", 8},     {"MnS", 9},  {"Na2S", 10},   {"ZnS", 11},
        {"KCl", 12},   {"NaCl", 13}, {"NH4Cl", 14}, {"H2O", 15}, {"NH3", 16},    {"CH4", 17},
        {"NH4SH", 18}, {"H2S", 19},  {"S2", 20},    {"S8", 21}};

    // Initialize the list of species that are active based on the parameter file
    for (const auto &name : sp) {
        if (species_map.find(name) != species_map.end()) {
            active_species.push_back(species_map[name]);
            std::cout << "Mapped species " << name << " to ID " << species_map[name] << std::endl;
        }
        else {
            std::cout << "Warning: Species " << name << " not found in species_map." << std::endl;
        }
    }

    // Print the size of active_species to verify the result of mapping
    std::cout << "Number of active species: " << active_species.size() << std::endl;

    // Print contents of active_species vector
    std::cout << "Active species IDs: ";
    for (const auto &id_spec : active_species) {
        std::cout << id_spec << " ";
    }
    std::cout << std::endl;

    cudaMalloc((void **)&active_species_d, active_species.size() * sizeof(int));

    // Copy active species identifiers to the device
    cudaMemcpy(active_species_d,
               active_species.data(),
               active_species.size() * sizeof(int),
               cudaMemcpyHostToDevice);

    // Register the arrays that need to be updated by the Runge-Kutta (RK) kernel
    phy_modules_core_arrays.register_array(
        tracers_cloud_d, tracerk_cloud_d, tracer_cloud_d, ntr_cloud);

#ifdef BENCHMARKING
    // Define variables for benchmarking and diagnostics
    std::map<std::string, output_def> defs = {
        {"tracer_cloud_d",
         {tracer_cloud_d, esp.nv * esp.point_num * ntr_cloud, "RK tracer", "ti", true}},
        {"tracers_cloud_d",
         {tracers_cloud_d, esp.nv * esp.point_num * ntr_cloud, "RK tracers", "ts", true}},
        {"tracerk_cloud_d",
         {tracerk_cloud_d, esp.nv * esp.point_num * ntr_cloud, "RK tracerk", "tk", true}},
        {"difftr_cloud_d",
         {difftr_cloud_d, esp.nv * esp.point_num * ntr_cloud, "Diffusion Tr", "dftr", true}}};

    // Specify which variables to output or monitor
    std::vector<std::string> output_vars = {
        "tracer_cloud_d", "tracers_cloud_d", "tracerk_cloud_d", "difftr_cloud_d"};

    // Register variables for benchmarking
    binary_test::get_instance().register_phy_modules_variables(
        defs, std::vector<std::string>(), output_vars);
#endif // BENCHMARKING

    return true;
}


bool clouds::free_memory() {
    // Free host memory
    free(tracer_cloud_h);

    // Free device memory
    cudaFree(tracer_cloud_d);
    cudaFree(tracers_cloud_d);
    cudaFree(tracerk_cloud_d);
    cudaFree(difftr_cloud_d);
    cudaFree(Kzzh_d);
    cudaFree(Rhoh_d);
    cudaFree(vf_d);
    cudaFree(vfh_d);
    cudaFree(y_old_d);
    cudaFree(y_stage_d);
    cudaFree(y_d);
    cudaFree(f_d);
    cudaFree(k1_d);
    cudaFree(k2_d);
    cudaFree(nd_atm_d);
    cudaFree(active_species_d);
    cudaFree(rho_pd_d);
    cudaFree(mol_w_sp_d);

    return true;
}


bool clouds::initial_conditions(const ESP &esp, const SimulationSetup &sim, storage *s) {
    bool returnstatus = true;

    if (s != nullptr) {
        // load initialisation data from storage s
        returnstatus &= (*s).read_table_to_ptr(
            "/tracer_cloud", tracer_cloud_h, esp.nv * esp.point_num * ntr_cloud);
    }

    if ((!sim.rest && !returnstatus) || s == nullptr) {
        printf("Adding cloud outputs!");
        const double R_UNIV_th = 8.31446261815324; // universal gas constant in J / ( K mol )
        // Initialize tracer arrays
        for (int id = 0; id < esp.point_num; id++) {
            for (int lev = 0; lev < esp.nv; lev++) {
                for (int icloud = 0; icloud < n_cloud; icloud++) {
                    // Breaking down the indexing logic
                    const int STRIDE_ID  = esp.nv * ntr_cloud;
                    const int STRIDE_LEV = ntr_cloud;
                    int       base_var   = 2 * icloud;
                    // Saturation vapour pressure
                    double p_vap = p_vap_sp(active_species[icloud], esp.temperature_h[id * esp.nv + lev]);
                    // Molar mass [kg/mol]
                    double mu = R_UNIV_th / esp.Rd_h[id * esp.nv + lev];
                    // Conversion factor between MMR and VMR for cloud species.
                    double eps = mol_w_sp[icloud] / mu;
                    // Equilibrium vapour MMR
                    double q_s = (p_vap / esp.pressure_h[id * esp.nv + lev]) * eps;
                    double q_v_deep_mmr = q_v_deep[icloud] * eps;

                    // Vapour (idx 0)
                    if (q_s <  q_v_deep_mmr && q_s > 1e-30){
                        tracer_cloud_h[id * STRIDE_ID + lev * STRIDE_LEV + base_var + 0] = q_s * esp.Rho_h[id * esp.nv + lev];
                    }else if (q_s > q_v_deep_mmr){
                        tracer_cloud_h[id * STRIDE_ID + lev * STRIDE_LEV + base_var + 0] = q_v_deep_mmr * esp.Rho_h[id * esp.nv + lev];
                    }else{
                        tracer_cloud_h[id * STRIDE_ID + lev * STRIDE_LEV + base_var + 0] = 1e-30 * esp.Rho_h[id * esp.nv + lev];
                    }
                    
                    // Condensate (idx 1)
                    tracer_cloud_h[id * STRIDE_ID + lev * STRIDE_LEV + base_var + 1] = 1e-30 * esp.Rho_h[id * esp.nv + lev];
                }
            }
        }
    }

    cudaMemcpy(tracer_cloud_d,
               tracer_cloud_h,
               esp.nv * esp.point_num * ntr_cloud * sizeof(double),
               cudaMemcpyHostToDevice);
    cudaMemset(tracers_cloud_d, 0, sizeof(double) * esp.nv * esp.point_num * ntr_cloud);
    cudaMemset(tracerk_cloud_d, 0, sizeof(double) * esp.nv * esp.point_num * ntr_cloud);


    return returnstatus;
}

bool clouds::dyn_core_loop_init(const ESP &esp) {

    cudaMemcpy(tracerk_cloud_d,
               tracer_cloud_d,
               esp.point_num * esp.nv * ntr_cloud * sizeof(double),
               cudaMemcpyDeviceToDevice);

    cudaMemset(tracers_cloud_d, 0, sizeof(double) * esp.point_num * esp.nv * ntr_cloud);

    return true;
}

bool clouds::dyn_core_loop_slow_modes(const ESP             &esp,
                                      const SimulationSetup &sim,
                                      int                    nstep, // Step number
                                      double                 time_step) {
    const int LN = 16;                             // Size of the inner region side.
    dim3      NT(esp.nl_region, esp.nl_region, 1); // Number of threads in a block.
    dim3      NBPT(2, 1, ntr_cloud);               // Number of blocks. (POLES)
    dim3 NBTR(esp.nr, esp.nv, ntr_cloud); // Number of blocks in the diffusion routine for tracers.
    dim3 NBTRP(
        2, esp.nv, ntr_cloud); // Number of blocks in the diffusion routine for tracers. (POLES)

    if (sim.HyDiff) {
        // Tracers
        cudaMemset(esp.diff_d, 0, sizeof(double) * 6 * esp.point_num * esp.nv);
        cudaDeviceSynchronize();
        Tracer_Eq_Diffusion<LN, LN><<<NBTR, NT>>>(difftr_cloud_d,
                                                  esp.diff_d,
                                                  tracerk_cloud_d,
                                                  esp.Rhok_d,
                                                  esp.areasTr_d,
                                                  esp.nvecoa_d,
                                                  esp.nvecti_d,
                                                  esp.nvecte_d,
                                                  esp.Kdh4_d,
                                                  esp.Altitude_d,
                                                  sim.A,
                                                  esp.maps_d,
                                                  ntr_cloud, //
                                                  esp.nl_region,
                                                  0,
                                                  sim.DeepModel);

        Tracer_Eq_Diffusion_Poles<5><<<NBTRP, 1>>>(difftr_cloud_d,
                                                   esp.diff_d,
                                                   tracerk_cloud_d,
                                                   esp.Rhok_d,
                                                   esp.areasTr_d,
                                                   esp.nvecoa_d,
                                                   esp.nvecti_d,
                                                   esp.nvecte_d,
                                                   esp.Kdh4_d,
                                                   esp.Altitude_d,
                                                   esp.Altitudeh_d,
                                                   sim.A,
                                                   esp.point_local_d,
                                                   ntr_cloud,
                                                   esp.point_num,
                                                   0,
                                                   sim.DeepModel);
        cudaDeviceSynchronize();
        Tracer_Eq_Diffusion<LN, LN><<<NBTR, NT>>>(difftr_cloud_d,
                                                  esp.diff_d,
                                                  tracerk_cloud_d,
                                                  esp.Rhok_d,
                                                  esp.areasTr_d,
                                                  esp.nvecoa_d,
                                                  esp.nvecti_d,
                                                  esp.nvecte_d,
                                                  esp.Kdh4_d,
                                                  esp.Altitude_d,
                                                  sim.A,
                                                  esp.maps_d,
                                                  ntr_cloud, //
                                                  esp.nl_region,
                                                  1,
                                                  sim.DeepModel);

        Tracer_Eq_Diffusion_Poles<5><<<NBTRP, 1>>>(difftr_cloud_d,
                                                   esp.diff_d,
                                                   tracerk_cloud_d,
                                                   esp.Rhok_d,
                                                   esp.areasTr_d,
                                                   esp.nvecoa_d,
                                                   esp.nvecti_d,
                                                   esp.nvecte_d,
                                                   esp.Kdh4_d,
                                                   esp.Altitude_d,
                                                   esp.Altitudeh_d,
                                                   sim.A,
                                                   esp.point_local_d,
                                                   ntr_cloud,
                                                   esp.point_num,
                                                   1,
                                                   sim.DeepModel);
    }
    cudaDeviceSynchronize();

    return true;
}

bool clouds::dyn_core_loop_fast_modes(const ESP             &esp,
                                      const SimulationSetup &sim,
                                      int                    nstep, // Step number
                                      double                 times) {
    const int LN = 16;                             // Size of the inner region side.
    dim3      NT(esp.nl_region, esp.nl_region, 1); // Number of threads in a block.
    dim3      NBPT(2, 1, ntr_cloud);               // Number of blocks. (POLES)
    dim3 NBTR(esp.nr, esp.nv, ntr_cloud); // Number of blocks in the diffusion routine for tracers.
    dim3 NBTRP(
        2, esp.nv, ntr_cloud); // Number of blocks in the diffusion routine for tracers. (POLES)


    //
    // Tracer equation.
    cudaDeviceSynchronize();
    Tracer_Eq<LN, LN><<<NBTR, NT>>>(tracers_cloud_d,
                                    tracerk_cloud_d,
                                    esp.Rhos_d,
                                    esp.Rhok_d,
                                    esp.Mhs_d,
                                    esp.Mhk_d,
                                    esp.Whs_d,
                                    esp.Whk_d,
                                    difftr_cloud_d,
                                    esp.div_d,
                                    esp.Altitude_d,
                                    esp.Altitudeh_d,
                                    sim.A,
                                    times,
                                    esp.maps_d,
                                    ntr_cloud,
                                    esp.nl_region,
                                    sim.DeepModel);

    Tracer_Eq_Poles<6><<<NBPT, 1>>>(tracers_cloud_d,
                                    tracerk_cloud_d,
                                    esp.Rhos_d,
                                    esp.Rhok_d,
                                    esp.Mhs_d,
                                    esp.Mhk_d,
                                    esp.Whs_d,
                                    esp.Whk_d,
                                    difftr_cloud_d,
                                    esp.div_d,
                                    esp.Altitude_d,
                                    esp.Altitudeh_d,
                                    sim.A,
                                    times,
                                    esp.point_local_d,
                                    ntr_cloud,
                                    esp.point_num,
                                    esp.nv,
                                    sim.DeepModel);

    cudaDeviceSynchronize();
    return true;
}


bool clouds::dyn_core_loop_end(const ESP &esp) {
    cudaMemcpy(tracer_cloud_d,
               tracerk_cloud_d,
               esp.point_num * esp.nv * ntr_cloud * sizeof(double),
               cudaMemcpyDeviceToDevice);

    return true;
}


bool clouds::phy_loop(ESP                   &esp,
                      const SimulationSetup &sim,
                      kernel_diagnostics    &diag,
                      int                    nstep, // Step number
                      double                 time_step) {

    USE_BENCHMARK()
    const int NTH = 256;
    dim3      NBTR3((esp.point_num / NTH) + 1, esp.nv, n_cloud);
    dim3      NBTR2((esp.point_num / NTH) + 1, 1, n_cloud);
    dim3      NBTR((esp.point_num / NTH) + 1, 1, 1);

    BENCH_POINT_I(nstep, "phy_clouds_begin", (), ("tracer_cloud_d"))

    ////////////////////////////
    // Simple cloud modelling //
    ////////////////////////////
    
    cudaDeviceSynchronize();

    // layers2interfaces<<<NBTR, NTH>>>(esp.Rho_d,       // Density [m3 kg-1] - Cell-centers
    //                                  Rhoh_d,          // Density [m3 kg-1] - Interfaces
    //                                  esp.Kzz_d,
    //                                  Kzzh_d,
    //                                  esp.point_num,   // Number of columns
    //                                  esp.nv);         // Vertical levels)

    // cudaDeviceSynchronize();
    
    // vert_diff_exp_linear<<<NBTR2, NTH>>>(esp.Rho_d,       // Density [m3 kg-1] - Cell-centers
    //                                     Rhoh_d,          // Density [m3 kg-1] - Interfaces
    //                                     esp.Altitude_d,  // Altitudes of the layers
    //                                     esp.Altitudeh_d, // Altitudes of the interfaces
    //                                     tracer_cloud_d,  // Array containing tracer VMRs
    //                                     esp.Kzz_d,
    //                                     Kzzh_d,
    //                                     y_d,             // Work arrays
    //                                     f_d,
    //                                     y_old_d,
    //                                     y_stage_d,
    //                                     k1_d,
    //                                     k2_d,
    //                                     sim.Gravit,      // Gravity [m s-2]
    //                                     time_step,       // time step [s]
    //                                     esp.point_num,   // Number of columns
    //                                     esp.nv);         // Vertical levels


    // cudaDeviceSynchronize();

    // Rounds thread number (number of columns) to the next multiple of 32
    int threads_per_block = esp.nv + 32 - esp.nv % 32;
    dim3 blockDim(threads_per_block, 1, 1);
    dim3 gridDim (esp.point_num, 1, n_cloud);
    
    // Shared memory array for parallelizing calculations on a column
    size_t shmem = 6 * threads_per_block * sizeof(double);   
    vert_diff_exp_parallel<<<gridDim, blockDim, shmem>>>(esp.Rho_d,       // Density [m3 kg-1] - Cell-centers
                                                        esp.Altitude_d,  // Altitudes of the layers
                                                        esp.Altitudeh_d, // Altitudes of the interfaces
                                                        tracer_cloud_d,  // Array containing tracer VMRs
                                                        esp.Kzz_d,
                                                        sim.Gravit,      // Gravity [m s-2]
                                                        time_step,       // time step [s]
                                                        esp.point_num,   // Number of columns
                                                        esp.nv);         // Vertical levels


    cudaDeviceSynchronize();

    mini_cloud<<<NBTR3, NTH>>>(esp.pressure_d,    // Pressure (cell centers) [Pa]
                               esp.temperature_d, // Temperature (cell centers)[K]
                               esp.Rho_d,         // Density [m3 kg-1]
                               esp.Cp_d,          // Specific heat capacity [J kg-1 K-1]
                               esp.Rd_d,          // Gas constant [J kg-1 K-1]
                               sim.Gravit,        // Gravity [m s-2]
                               tau_deep_d,        // Deep replenishment timescale [s]
                               tracer_cloud_d,    // Array containing tracer MMRs
                               q_v_deep_d,        // Deep vapor mixing ratio (MMR)
                               vf_d,              // Settling velocity [m s-1]
                               tau_chem_d,        // Equilibrium cloud timescale [s]
                               rho_pd_d,          // Particle density [kg m-3]
                               rm_d,              // Median particle size [m]
                               active_species_d,  // Integer identifiers for active cloud species
                               mol_w_sp_d,        // Molecular weight of species [kg mol-1]
                               y_d,               // Work arrays
                               f_d,
                               time_step,     // time step [s]
                               ntr_cloud,     // Number of cloud tracers
                               esp.point_num, // Number of columns
                               sim.GravHeightVar);

    // cudaDeviceSynchronize();

    mini_cloud_settling_velocity<<<NBTR3, NTH>>>(esp.pressure_d,    // Pressure (cell centers) [Pa]
                                                 esp.temperature_d, // Temperature (cell centers)[K]
                                                 esp.Rho_d,         // Density [m3 kg-1]
                                                 esp.Cp_d,        // Specific heat capacity [J kg-1 K-1]
                                                 esp.Rd_d,        // Gas constant [J/kg/K]
                                                 sim.Gravit,      // Gravity [m s-2]
                                                 esp.Altitude_d,  // Altitudes of the layers
                                                 esp.Altitudeh_d, // Altitudes of the interfaces
                                                 vf_d, // Gravitational settling velocity [m s-1]
                                                 nd_atm_d,
                                                 rho_pd_d,       // Particle bulk density [kg/m3]
                                                 sigma_d,        // Particle size distribution std
                                                 rm_d,           // Median particle size [m]
                                                 time_step,      // time step [s]
                                                 esp.point_num); // Number of columns

    // cudaDeviceSynchronize();

    vert_adv_exp_maccormack<<<NBTR, NTH>>>(esp.Rho_d,       // Density [m3 kg-1]
                                           esp.Altitude_d,  // Altitudes of the layers
                                           esp.Altitudeh_d, // Altitudes of the interfaces
                                           tracer_cloud_d,  // Array containing tracer VMRs
                                           vf_d,            // Gravitational settling velocity [m s-1] - Cell-centers
                                           vfh_d,           // Gravitational settling velocity [m s-1] - Interfaces
                                           nd_atm_d,
                                           sim.Gravit,    // Gravity [m s-2]
                                           time_step,     // time step [s]
                                           ntr_cloud,     // Number of cloud tracers
                                           n_cloud,       // Number of cloud species
                                           esp.point_num, // Number of columns
                                           esp.nv);       // Vertical levels

    // cudaDeviceSynchronize();
    
    cloud_number_density<<<NBTR3, NTH>>>(esp.Rho_d,      // Density [m3 kg-1]
                                         esp.Rd_d,       // Gas constant [J kg-1 K-1]
                                         tracer_cloud_d, // Array containing tracer VMRs
                                         mol_w_sp_d,     // Molecular weight of species [kg mol-1]
                                         esp.n_tot_d,    // Total number density
                                         rho_pd_d,       // Particle bulk density [kg m-3]
                                         sigma_d,        // Particle size distribution std
                                         rm_d,           // Median particle size [m]
                                         ntr_cloud,      // Number of cloud tracers
                                         esp.point_num); // Number of columns

    cudaDeviceSynchronize();

    BENCH_POINT_I(nstep, "phy_clouds_end", (), ("tracer_cloud_d"))

    return true;
}

bool clouds::configure(config_file &config_reader) {

    // Cloud tracer specifications
    config_reader.append_config_var("n_cloud", n_cloud, n_cloud);
    config_reader.append_config_var("sp", sp, sp);

    // Physical properties for the cloud species
    config_reader.append_config_var("tau_deep", tau_deep_d, tau_deep_d);
    config_reader.append_config_var("q_v_deep", q_v_deep, q_v_deep);
    config_reader.append_config_var("tau_chem", tau_chem_d, tau_chem_d);
    config_reader.append_config_var("rho_pd", rho_pd, rho_pd);
    config_reader.append_config_var("sigma", sigma_d, sigma_d);
    config_reader.append_config_var("rm", rm_d, rm_d);
    config_reader.append_config_var("mol_w_sp", mol_w_sp, mol_w_sp);
    return true;
}

bool clouds::store(const ESP &esp, storage &s) {
    // Add tracers to the output
    cudaMemcpy(tracer_cloud_h,
               tracer_cloud_d,
               esp.point_num * esp.nv * ntr_cloud * sizeof(double),
               cudaMemcpyDeviceToHost);
    s.append_table(tracer_cloud_h,
                   esp.nv * esp.point_num * ntr_cloud,
                   "/tracer_cloud",
                   " ",
                   "Mass-mixing ratio");

    return true;
}

bool clouds::store_init(storage &s) {
    return true;
}
