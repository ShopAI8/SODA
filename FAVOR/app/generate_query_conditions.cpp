#include <iostream>
#include <fstream>
#include <random>
#include <string>
#include <vector>
#include <set>

#include "dataset.h"

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " queryset_path" << " condition_path\n";
        return 1;
    }

    std::string queryset_path(argv[1]);
    std::string condition_path(argv[2]);

    QuerySet queryset;
    queryset.read_data(queryset_path);

    std::ofstream outFile(condition_path);
    if (!outFile)
    {
        std::cerr << "Error: Unable to create file at " << condition_path << std::endl;
        return 1;
    }

    std::cout << "Generating filtering conditions for " << queryset.num << " queries" << std::endl;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> opDist(0, 3);
    std::uniform_int_distribution<> valueDist(0, 99);
    std::uniform_int_distribution<> inListSizeDist(1, 5);

    for (int i = 0; i < queryset.num; ++i)
    {
        int opType = opDist(gen);
        
        if (opType == 0)
        {
            int value = valueDist(gen);
            outFile << "color == " << value << "\n";
        }
        else if (opType == 1)
        {
            int value = valueDist(gen);
            outFile << "color != " << value << "\n";
        }
        else if (opType == 2)
        {
            std::set<int> values;
            int listSize = inListSizeDist(gen);
            while (values.size() < static_cast<size_t>(listSize))
            {
                values.insert(valueDist(gen));
            }
            
            outFile << "color IN [";
            bool first = true;
            for (int val : values)
            {
                if (!first) outFile << ", ";
                outFile << val;
                first = false;
            }
            outFile << "]\n";
        }
        else
        {
            int opType2 = std::uniform_int_distribution<>(0, 1)(gen);
            int value = valueDist(gen);
            if (opType2 == 0)
            {
                outFile << "color > " << value << "\n";
            }
            else
            {
                outFile << "color < " << value << "\n";
            }
        }
    }

    outFile.close();
    std::cout << "Filtering conditions successfully generated at: " << condition_path << std::endl;
    return 0;
}
