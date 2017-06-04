
/* 
 * File:   csv_writer.h
 * Author: ryan
 *
 * Created on 03 June 2017, 14:37
 */

#ifndef CSV_WRITER_H
#define CSV_WRITER_H

#include "graphs/dnn_graph_settings.h"

#include <Eigen/Dense>
#include <iostream>

using namespace Eigen;

class CSVWriter{
public:
    template<typename TMat>
    static void writeCSV(string path, TMat mat){
        Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");
        ofstream file(path.c_str());
        file << mat.format(CSVFormat);
    }

    template<typename TVec, typename TMat>    
    static void writeCSV(string path, TVec mat1, TVec mat2, TVec mat3){
        Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");
        ofstream file(path.c_str());
        TMat mat(mat1.size(),3);
        mat << mat1, mat2, mat3;
        file << mat.format(CSVFormat);
    }
};

#endif /* CSV_WRITER_H */

