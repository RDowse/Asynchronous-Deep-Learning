
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
#include <fstream>

using namespace Eigen;

class CSVWriter{
public:
    template<typename TMat>
    static void writeCSV(string path, TMat mat, string headings=""){
        cout << "Writing CSV: " << path << endl;
        Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");
        ofstream file(path.c_str());
        file << mat.format(CSVFormat);
        file.close();
    }

    template<typename TVec, typename TMat>    
    static void writeCSV(string path, TVec mat1, TVec mat2, TVec mat3, string headings=""){
        cout << "Writing CSV: " << path << endl;
        Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");
        ofstream file(path.c_str());
        TMat mat(mat1.size(),3);
        mat << mat1, mat2, mat3;
        file << mat.format(CSVFormat);
        file.close();
    }
    
    static void writeContext(string path, shared_ptr<DNNGraphSettings> context){
        cout << "Writing config: " << path << endl;
        ofstream file(path.c_str());
        file << "DNNSettings\n";
        file << "NetType: " << context->netType << endl << endl;
        
        file << "Topology\n";
        file << "Num Hidden Layers: " << context->nHLayers << endl;
        file << "Num Hidden Per Layer: " << context->nHidden << endl;
        file << "Num Input: " << context->nInput << endl;
        file << "Num output: " << context->nOutput << endl << endl;
        
        file << "Training Parameters\n";
        file << "lr: " << context->lr << endl;
        file << "alpha: " << context->alpha << endl;
        file << "batchSize: " << context->batchSize << endl;
        file << "maxEpoch: " << context->maxEpoch << endl << endl;
        
        if("async_neural" == context->netType){
            file << "Edge Delays" << endl;
            file << "EnableEdgeDelay: " << context->enableVariableEdgeDelay << endl;
            file << "Wait time: "<< context->waitTime << endl;
            file << "Wait time factor: "<< context->waitTimeFactor << endl;
            file << "Mean: "<< context->mean << endl;
            file << "Std: "<< context->std << endl;
            file << "Forward drop tolerance: " << context->forwardDropTolerance << endl;
            file << "Backward drop tolerance: " << context->backwardDropTolerance << endl << endl;
            
            file << "Message Stats" << endl;
            file << "NumForwardMessagesSent: " << context->numForwardMessagesSent << endl;
            file << "NumBackwardMessagesSent: " << context->numBackwardMessagesSent << endl;
            file << "NumForwardMessagesDropped: " << context->numForwardMessagesDropped << endl;
            file << "NumBackwardMesssagesDropped: " << context->numBackwardMessagesDropped << endl << endl;
            
            file << "NumForwardMessagesSentSync: " << context->numForwardMessagesSentSync << endl;
            file << "NumBackwardMessagesSentSync: " << context->numBackwardMessagesSentSync << endl;
        }
        
        file << "Runtime: " << context->runTime << endl;
        file << "TotalSimSteps: " << context->stepTime << endl;
    }
};

#endif /* CSV_WRITER_H */

