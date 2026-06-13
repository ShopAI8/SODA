#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

#include "dataset.h"

void BaseSet::get_attribute(const std::string &attribute_path)
{
    std::ifstream infile(attribute_path);
    if (!infile.is_open()) {
        throw std::runtime_error("Error: Could not open file " + attribute_path);
    }

    attribute_map.clear();
    if (attribute != nullptr) {
        delete[] attribute;
        attribute = nullptr;
    }

    std::string line;

    if (!std::getline(infile, line)) {
        throw std::runtime_error("Error: Missing num value in the first line");
    }
    try {
        num = std::stoi(line);
    } catch (...) {
        throw std::runtime_error("Error: Invalid num value in the first line: '" + line + "'");
    }

    if (!std::getline(infile, line)) {
        throw std::runtime_error("Error: Missing num value in the second line");
    }
    try {
        attribute_num = std::stoi(line);
    } catch (...) {
        throw std::runtime_error("Error: Invalid attribute_num value in the second line: '" + line + "'");
    }

    attribute = new float[attribute_num * num];
    
    for (int i = 0; i < attribute_num; ++i) {
        if (!std::getline(infile, line)) {
            delete[] attribute;
            attribute = nullptr;
            throw std::runtime_error("Error: Unexpected end of file at attribute string for attribute " + 
                                    std::to_string(i));
        }
        
        while (line.empty()) {
            if (!std::getline(infile, line)) {
                delete[] attribute;
                attribute = nullptr;
                throw std::runtime_error("Error: Unexpected end of file while skipping empty lines for attribute " +
                                       std::to_string(i));
            }
        }

        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }

        attribute_map.insert({line, i});

        
        for (int j = 0; j < num; ++j) {
            if (!std::getline(infile, line)) {
                delete[] attribute;
                attribute = nullptr;
                throw std::runtime_error("Error: Unexpected end of file at float data for attribute " + 
                                       std::to_string(i) + ", position " + std::to_string(j));
            }
            
            while (line.empty()) {
                if (!std::getline(infile, line)) {
                    delete[] attribute;
                    attribute = nullptr;
                    throw std::runtime_error("Error: Unexpected end of file while skipping empty lines for attribute " +
                                           std::to_string(i) + ", position " + std::to_string(j));
                }
            }

            try {
                attribute[j*attribute_num+i] = std::stof(line);
            } catch (const std::exception& e) {
                delete[] attribute;
                attribute = nullptr;
                throw std::runtime_error("Error: Invalid float value at attribute " + 
                                       std::to_string(i) + ", position " + std::to_string(j) +
                                       ": '" + line + "' - " + e.what());
            }
        }
    }
    
    infile.close();
}