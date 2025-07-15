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
// Description: helper functions for parsers
//
//
// Known limitations: None.
//
//
// Known issues: None.
//
//
// Authors: Joao Mendonca, EEG. joao.mendonca@csh.unibe.ch
//          Urs Schroffenegger, Russell Deitrick
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
#pragma once

#include <string>

#include <iostream>

#include <sstream>
#include <vector>


// To compile without regex lib
//#define NO_REGEX_SUPPORT

using std::string;


// check if we use GCC and if version is below 4.9, define
// NO_REGEX_SUPPORT
#ifdef __GNUC__
#    if ((__GNUC__ == 4 && __GNUC_MINOR__ < 9) || (__GNUC__ < 4))
#        define NO_REGEX_SUPPORT
#    endif // GCC version check
#endif     // __GNUC__

#ifdef NO_REGEX_SUPPORT
#else // NO_REGEX_SUPPORT
#    include <regex>

using std::regex;

#endif // NO_REGEX_SUPPORT

inline bool is_empty_line(const string& line) {
#ifdef NO_REGEX_SUPPORT
    bool is_empty = true;

    for (size_t i = 0; i < line.length(); i++) {
        // is it a whitespace character
        is_empty &= (line[i] == ' ' || line[i] == '\t');
    }

    return is_empty;


#else  // NO_REGEX_SUPPORT
    std::smatch match;
    regex       empty_line_regex("^[[:blank:]]*$");
    return regex_match(line, match, empty_line_regex);
#endif // NO_REGEX_SUPPORT
}


inline bool is_comment_line(const string& line) {
#ifdef NO_REGEX_SUPPORT
    bool is_empty = true;

    for (size_t i = 0; i < line.length(); i++) {
        // is it a whitespace character
        is_empty &= (line[i] == ' ' || line[i] == '\t');


        if (!is_empty) {
            if (line[i] == '#')
                return true;
            else
                return false;
        }
    }

    return is_empty;
#else // NO_REGEX_SUPPORT
    std::smatch match;
    regex       comment_regex("^[[:blank:]]?#.*$");

    return regex_match(line, match, comment_regex);

#endif // NO_REGEX_SUPPORT
}

inline bool is_key_value_pair(const string& line, string& key, string& value) {
#ifdef NO_REGEX_SUPPORT


    size_t i = 0;

    // strip whitespace
    for (; i < line.length(); i++) {
        // is it a whitespace character
        if (line[i] == ' ' || line[i] == '\t') {
            // ignore
        }
        else
            break;
    }
    if (i == line.length())
        return false;

    // append char as key until whitespace or = symbol
    key = "";
    for (; i < line.length(); i++) {
        if (line[i] == ' ' || line[i] == '\t' || line[i] == '=') {
            // end of key, leave
            break;
        }
        else
            key += line[i];
    }

    // go to equal symbol
    for (; i < line.length(); i++) {
        if (line[i] == ' ' || line[i] == '\t') {
            // continue
        }
        else if (line[i] == '=') {
            break;
        }
        else
            return false;
    }

    if (i == line.length())
        return false;

    i++;

    // strip whitespace
    for (; i < line.length(); i++) {
        // is it a whitespace character
        if (line[i] == ' ' || line[i] == '\t') {
            // ignore
        }
        else
            break;
    }
    if (i == line.length())
        return false;

    // append char as value until whitespace or = symbol
    value = "";
    for (; i < line.length(); i++) {
        if (line[i] == ' ' || line[i] == '\t' || line[i] == '#') {
            // end of value, leave
            break;
        }
        else
            value += line[i];
    }
    if (value.length() > 0)
        return true;

    return false;


#else  // NO_REGEX_SUPPORT
    std::smatch match;

    regex key_value_pair_regex("^[[:blank:]]*([A-Za-z0-9_]+)[[:blank:]]*="
                               "[[:blank:]]*(([[:alnum:]]|[[:punct:]])+)[[:blank:]]*(#.*)?$");
    if (regex_match(line, match, key_value_pair_regex)) {
        key   = match[1];
        value = match[2];

        return true;
    }
    else
        return false;
#endif // NO_REGEX_SUPPORT
}


inline bool is_long_cmdargs(const string& in, string& out) {

#ifdef NO_REGEX_SUPPORT
    if (in.length() >= 3 and in[0] == '-' and in[1] == '-') {
        for (size_t i = 2; i < in.length(); i++) {
            if ((in[i] == ' ') || (in[i] == '-') || (in[i] == '\t'))
                return false;
            out += in[i];
        }
        return true;
    }
    return false;

#else  // NO_REGEX_SUPPORT
    regex long_key_regex("^--([A-Za-z0-9]+)$");

    std::smatch match;

    if (regex_match(in, match, long_key_regex)) {
        out = match[1];
        return true;
    }
    else
        return false;
#endif // NO_REGEX_SUPPORT
}


inline bool is_short_cmdargs(const string& in, string& out) {
#ifdef NO_REGEX_SUPPORT
    if (in.length() == 2 and in[0] == '-') {
        out = in.substr(1, string::npos);

        return (out[0] != ' ') && (out[0] != '-') && (out[0] != '\t');
    }
    return false;

#else  // NO_REGEX_SUPPORT
    regex       short_key_regex("^-([A-Za-z0-9])$");
    std::smatch match;

    if (regex_match(in, match, short_key_regex)) {
        out = match[1];
        return true;
    }
    else
        return false;
#endif // NO_REGEX_SUPPORT
}

// Parsing functions to parse string to a value, on same
// interface. Functions should recognise the output type we want to
// parse to from their argument.
// Used by config file parser and argument parser

// base function prototype
template<typename T> bool parse_data(const string& value, T& target) {

    return false;
}

// bool template specialisation
template<> inline bool parse_data(const string& value, bool& target) {
#ifdef NO_REGEX_SUPPORT
    if (value == string("true")) {
        target = true;
        return true;
    }
    else if (value == string("false")) {
        target = false;
        return true;
    }
#else  //  NO_REGEX_SUPPORT
    regex bool_regex("^(true|false)$");

    std::smatch match;

    if (regex_match(value, match, bool_regex)) {
        target = (match[1] == "true");
        // cout << "parsed [" << value << "] as [" << target << "]" << endl;

        return true;
    }
#endif // NO_REGEX_SUPPORT

    return false;
}

// int template specialisation
template<> inline bool parse_data(const string& value, int& target) {
#ifdef NO_REGEX_SUPPORT
    try {
        target = std::stoi(value);

        return true;
    } catch (...) {
        return false;
    }

#else  //  NO_REGEX_SUPPORT
    regex int_regex("^((-|\\+)?[0-9]+)$");

    std::smatch match;

    if (regex_match(value, match, int_regex)) {
        target = std::stoi(match[1]);
        // cout << "parsed [" << value << "] as [" << target << "]" << endl;

        return true;
    }

    return false;
#endif // NO_REGEX_SUPPORT
}

// double template specialisation
template<> inline bool parse_data(const string& value, double& target) {
#ifdef NO_REGEX_SUPPORT
    try {
        target = std::stod(value);
        return true;
    } catch (...) {
        std::cerr << "Failed to parse double: [" << value << "]" << std::endl; // Debugging output
        return false;
    }

#else  //  NO_REGEX_SUPPORT
    regex double_regex("^((-|\\+)?[0-9]+(\\.[0-9]+)?((E|e)(-|\\+)?[0-9]+)?)$");

    std::smatch match;

    if (regex_match(value, match, double_regex)) {
        target = std::stod(match[1]);
        //cout << "parsed [" << value << "] as [" << target << "]" << endl;

        return true;
    }

    return false;
#endif // NO_REGEX_SUPPORT
}

// string template specialisation
template<> inline bool parse_data(const string& value, string& target) {
    target = value;
    std::cout << "Parsed string value: [" << target << "]" << std::endl; // Debugging output
    return true;
}

template<> inline bool parse_data(const std::string& value, std::vector<std::string>& target) {
    target.clear();
    std::istringstream ss(value);
    std::string item;

    // Split on commas and remove any extra whitespace from each item
    while (std::getline(ss, item, ',')) {
        item.erase(0, item.find_first_not_of(" \t"));  // Left trim
        item.erase(item.find_last_not_of(" \t") + 1);  // Right trim
        target.push_back(item);
    }

    // Debug output for verification
    std::cout << "Parsed vector<string> successfully: [";
    for (const auto& str : target) {
        std::cout << str << " ";
    }
    std::cout << "]" << std::endl;

    return !target.empty();  // Return true if at least one element was parsed
}

template<> inline bool parse_data(const std::string& value, std::vector<double>& target) {
    target.clear();
    std::istringstream ss(value);
    std::string item;

    try {
        while (std::getline(ss, item, ',')) {
            // Trim whitespace from each item
            item.erase(0, item.find_first_not_of(" \t"));
            item.erase(item.find_last_not_of(" \t") + 1);
            target.push_back(std::stod(item));
        }
        std::cout << "Parsed vector<double> successfully: [";
        for (const auto& num : target) {
            std::cout << num << " ";
        }
        std::cout << "]" << std::endl;
        return !target.empty();
    } catch (const std::exception& e) {
        std::cerr << "Failed to parse vector<double>: [" << value << "], error: " << e.what() << std::endl;
        return false;
    }
}

// Conversion from value to string, recognising input type
// template<typename T> inline string to_strg(T& val) {
//     return std::to_string(val);
// }
// template<> inline string to_strg(string& val) {
//     return val;
// }
// template<> inline string to_strg(bool& val) {
//     return val ? "1" : "0";
// }

// Base template function for to_strg, using std::to_string for scalars
template<typename T> inline std::string to_strg(T& val) {
    return std::to_string(val);
}

// Specialization for std::string type
template<> inline std::string to_strg(std::string& val) {
    return val;
}

// Specialization for bool type
template<> inline std::string to_strg(bool& val) {
    return val ? "1" : "0";
}

// Specialization for std::vector<T> to handle vectors (e.g., vector<string> and vector<double>)
template<typename T> inline std::string to_strg(std::vector<T>& vec) {
    std::ostringstream oss;
    for (size_t i = 0; i < vec.size(); ++i) {
        oss << to_strg(vec[i]); // Recursively call to_strg for each element
        if (i != vec.size() - 1) {
            oss << ", ";
        }
    }
    return oss.str();
}

// helpers to check coherence of input
template<typename T> bool check_greater(string name, T val, T min_) {
    if (val > min_)
        return true;
    else {
        std::cout << "Bad range for " << name << " (cond: min(" << min_ << ") < " << val
                  << std::endl;

        return false;
    }
}


template<typename T> bool check_range(const string& name, T val, T min_, T max_) {
    if (val > min_ && val < max_)
        return true;
    else {
        std::cout << "Bad range for " << name << " (cond: min(" << min_ << ") < " << val
                  << ") < max(" << max_ << "))" << std::endl;

        return false;
    }
}
