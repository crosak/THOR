#pragma once

#include <fstream>
#include <iostream>
#include <string>

///////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////


// Calculates the Bond Albedo according to Parmentier et al. (2015) expression
void Bond_Parmentier(double Teff0, double grav, double &AB) {
    // dependcies
    //// pow from math
    //// log10 from math


    // Input:
    // Teff0 - Atmospheric profile effective temperature [K] with zero albedo
    // grav - Surface gravity of planet [m s-2]

    // Call by reference (Input&Output):
    // AB - Bond albedo

    // work variables
    double a = 0.0, b = 0.0;

    // start operations

    if (Teff0 <= 250.0) {
        a = ((double)-0.335) * pow(grav, ((double)0.070));
        b = 0.0;
    }
    else if (Teff0 > 250.0 && Teff0 <= 750.0) {
        a = -0.335 * pow(grav, ((double)0.070)) + 2.149 * pow(grav, ((double)0.135));
        b = -0.896 * pow(grav, ((double)0.135));
    }
    else if (Teff0 > 750.0 && Teff0 < 1250.0) {
        a = -0.335 * pow(grav, ((double)0.070)) - 0.428 * pow(grav, ((double)0.135));
        b = 0.0;
    }
    else if (Teff0 >= 1250.0) {
        a = 16.947 - ((double)3.174) * pow(grav, ((double)0.070))
            - 4.051 * pow(grav, ((double)0.135));
        b = -5.472 + ((double)0.917) * pow(grav, ((double)0.070))
            + 1.170 * pow(grav, ((double)0.135));
    }

    // Final Bond Albedo expression
    AB = pow(10.0, (a + b * log10(Teff0)));
}


///////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////


void PF_text_file_to_array(std::string name, double *array, int Nlength) {

    std::ifstream inFile;
    inFile.open(name);
    if (!inFile) {
        printf("\nError opening the file: one of the opacity tables \n");
    }
    for (int i = 0; i < Nlength; i++) {
        inFile >> array[i];
    }
    inFile.close();
}


///////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

// Calculates 3 band grey visual gamma values and 2 picket fence IR gamma values
// according to the coefficents and equations in:
// Parmentier & Menou (2014) and Parmentier et al. (2015)
// NOTE: This does not calculate the opacity - call k_Ross_Freedman for that
void gam_Parmentier(int     nCol,
                    int     nLev,
                    double *Teff,
                    int     table_num,
                    double *gam_V,
                    double *Beta_V,
                    double *Beta,
                    double *gam_1,
                    double *gam_2,
                    double *gam_P) {
    // dependcies
    //// pow from math
    //// log10 from math


    // Input:
    // Teff - Effective temperature [K] (See Parmentier papers for various ways to calculate this)
    // for non-irradiated atmosphere Teff = Tint
    // table_num - Table selection from Parmentier et al. (2015): 1 = w. TiO/VO, 2 = w.o. TiO/VO

    // Call by reference (Input&Output):
    // gam_V(3) - gamma ratio for 3 visual bands (gam_V = kV_Ross/kIR_Ross)
    // beta_V(3) - fraction of total incident stellar flux in band (1/3 for Parmentier values)
    // Beta - equilvalent bandwidth for picket fence IR model
    // gam_1 - gamma ratio for IR band 1 (gam_1 = kIR_1/kIR_Ross)
    // gam_2 - gamma ratio for IR band 2 (gam_2 = kIR_2/kIR_Ross)
    // gam_P - gamma ratio for Planck mean (gam_P = kIR_Planck/kIR_Ross)
    // tau_lim - tau limit variable (usually for IC system)

    // work variables
    double R   = 0;
    double aP  = 0;
    double bP  = 0;
    double cP  = 0;
    double aV1 = 0, bV1 = 0, aV2 = 0, bV2 = 0, aV3 = 0, bV3 = 0;
    double aB = 0, bB = 0;
    double l10T = 0, l10T2 = 0, RT = 0;
    int    i;


    // start operations

    for (int id = 0; id < nCol; id++) {

        // Log 10 T_eff variables
        l10T  = log10(Teff[id]);
        l10T2 = pow(l10T, 2.0);

        if (table_num == 1) {
            // First table in Parmentier et al. (2015) w. TiO/VO
            // Start large if statements with visual band and Beta coefficents
            if (Teff[id] <= 200.0) {
                aV1 = -5.51;
                bV1 = 2.48;
                aV2 = -7.37;
                bV2 = 2.53;
                aV3 = -3.03;
                bV3 = -0.20;
                aB  = 0.84;
                bB  = 0.0;
            }
            else if (Teff[id] > 200.0 && Teff[id] <= 300.0) {
                aV1 = 1.23;
                bV1 = -0.45;
                aV2 = 13.99;
                bV2 = -6.75;
                aV3 = -13.87;
                bV3 = 4.51;
                aB  = 0.84;
                bB  = 0.0;
            }
            else if (Teff[id] > 300.0 && Teff[id] <= 600.0) {
                aV1 = 8.65;
                bV1 = -3.45;
                aV2 = -15.18;
                bV2 = 5.02;
                aV3 = -11.95;
                bV3 = 3.74;
                aB  = 0.84;
                bB  = 0.0;
            }
            else if (Teff[id] > 600.0 && Teff[id] <= 1400.0) {
                aV1 = -12.96;
                bV1 = 4.33;
                aV2 = -10.41;
                bV2 = 3.31;
                aV3 = -6.97;
                bV3 = 1.94;
                aB  = 0.84;
                bB  = 0.0;
            }
            else if (Teff[id] > 1400.0 && Teff[id] < 2000.0) {
                aV1 = -23.75;
                bV1 = 7.76;
                aV2 = -19.95;
                bV2 = 6.34;
                aV3 = -3.65;
                bV3 = 0.89;
                aB  = 0.84;
                bB  = 0.0;
            }
            else if (Teff[id] >= 2000.0) {
                aV1 = 12.65;
                bV1 = -3.27;
                aV2 = 13.56;
                bV2 = -3.81;
                aV3 = -6.02;
                bV3 = 1.61;
                aB  = 6.21;
                bB  = -1.63;
            }

            // gam_P coefficents
            aP = -2.36;
            bP = 13.92;
            cP = -19.38;
        }
        else if (table_num == 2) {
            // ! Appendix table from Parmentier et al. (2015) - without TiO and VO
            if (Teff[id] <= 200.0) {
                aV1 = -5.51;
                bV1 = 2.48;
                aV2 = -7.37;
                bV2 = 2.53;
                aV3 = -3.03;
                bV3 = -0.20;
                aB  = 0.84;
                bB  = 0.0;
            }
            else if (Teff[id] > 200.0 && Teff[id] <= 300.0) {
                aV1 = 1.23;
                bV1 = -0.45;
                aV2 = 13.99;
                bV2 = -6.75;
                aV3 = -13.87;
                bV3 = 4.51;
                aB  = 0.84;
                bB  = 0.0;
            }
            else if (Teff[id] > 300.0 && Teff[id] <= 600.0) {
                aV1 = 8.65;
                bV1 = -3.45;
                aV2 = -15.18;
                bV2 = 5.02;
                aV3 = -11.95;
                bV3 = 3.74;
                aB  = 0.84;
                bB  = 0.0;
            }
            else if (Teff[id] > 600.0 && Teff[id] <= 1400.0) {
                aV1 = -12.96;
                bV1 = 4.33;
                aV2 = -10.41;
                bV2 = 3.31;
                aV3 = -6.97;
                bV3 = 1.94;
                aB  = 0.84;
                bB  = 0.0;
            }
            else if (Teff[id] > 1400.0 && Teff[id] < 2000.0) {
                aV1 = -1.68;
                bV1 = 0.75;
                aV2 = 6.96;
                bV2 = -2.21;
                aV3 = 0.02;
                bV3 = -0.28;
                aB  = 3.0;
                bB  = -0.69;
            }
            else if (Teff[id] >= 2000.0) {
                aV1 = 10.37;
                bV1 = -2.91;
                aV2 = -2.4;
                bV2 = 0.62;
                aV3 = -16.54;
                bV3 = 4.74;
                aB  = 3.0;
                bB  = -0.69;
            }

            // gam_P coefficents
            if (Teff[id] <= 1400.0) {
                aP = -2.36;
                bP = 13.92;
                cP = -19.38;
            }
            else {
                aP = -12.45;
                bP = 82.25;
                cP = -134.42;
            }
        }

        // Calculation of all values
        // Visual band gamma
        gam_V[id * 3 + 0] = pow(10.0, (aV1 + bV1 * l10T));
        gam_V[id * 3 + 1] = pow(10.0, (aV2 + bV2 * l10T));
        gam_V[id * 3 + 2] = pow(10.0, (aV3 + bV3 * l10T));


        // Visual band fractions
        for (i = 0; i < 3; i++) {
            Beta_V[id * 3 + i] = 1.0 / 3.0;
        }

        // gamma_Planck - if < 1 then make it grey approximation (k_Planck = k_Ross, gam_P = 1)
        gam_P[id] = pow(10.0, (aP * l10T2 + bP * l10T + cP));
        if (gam_P[id] < 1.0000001) {
            gam_P[id] = 1.0000001;
        }

        // equivalent bandwidth value
        Beta[id * 2 + 0] = aB + bB * l10T;
        Beta[id * 2 + 1] = (1.0) - Beta[id * 2 + 0];

        // IR band kappa1/kappa2 ratio - Eq. 96 from Parmentier & Menou (2014)
        RT = (gam_P[id] - 1.0) / (2.0 * Beta[id * 2 + 0] * Beta[id * 2 + 1]);
        R  = 1.0 + RT + sqrt(pow(RT, 2.0) + RT);

        // gam_1 and gam_2 values - Eq. 92, 93 from Parmentier & Menou (2014)
        gam_1[id] = Beta[id * 2 + 0] + R - Beta[id * 2 + 0] * R;
        gam_2[id] = gam_1[id] / R;

        // Calculate tau_lim parameter -> not anymore needed
        //tau_lim = 1.0_dp/(gam_1[id]*gam_2[id]) * sqrt(gam_P[id]/3.0_dp);
    }
}


///////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////


void read_cloud_tables(const std::string &filename, 
                       double* T_array, 
                       double* vals_array, 
                       int N)
{
    // printf("Reading file: %s \n", filename.c_str());
    std::ifstream inFile(filename);
    if (!inFile.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    // Skip the first header line
    {
        std::string dummy;
        std::getline(inFile, dummy);
    }

    // Skip the second header line (median size, etc.)
    {
        std::string dummy;
        std::getline(inFile, dummy);
    }

    // Read N temperature points from the third line
    for (int i = 0; i < N; i++) {
        inFile >> T_array[i];
    }

    // Read N data values (the property) from the fourth line
    for (int i = 0; i < N; i++) {
        inFile >> vals_array[i];
    }

    inFile.close();
}