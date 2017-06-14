
/* 
 * File:   yaml_reader.h
 * Author: ryan
 *
 * Created on 18 May 2017, 16:49
 */

#ifndef YAML_READER_H
#define YAML_READER_H

#include "graphs/dnn_graph_settings.h"

#include <yaml-cpp/yaml.h>
#include <locale>
#include <algorithm>
#include <iostream>
#include <random>
#include <memory>

using namespace std;

class YamlReader{
    YAML::Node config;
    std::string command;
public:
    YamlReader(std::string path){
        cout << "Reading yaml file: " << path << "\n";
        config = YAML::LoadFile(path);
        if(config["command"].IsDefined()){
            cout << "command: " << config["command"].as<string>() << "\n";
            command = toLower(config["command"].as<string>());
            if(command != "build" && command != "run"){
                cout << "Error: command '" << command << "' not recognised\n";
                usage();
                exit(1);
            }
        } else {
            cout << "No command given\n";
            exit(1);
        }
    }
    
    void readConfig(shared_ptr<DNNGraphSettings>& settings){
        if(command == "build"){
            build(settings);
        } else if(command == "run"){
            run(settings);
        } else{
            cout << "Unknown command: " << command << "\n";
            assert(0);
        }
    }
    
private:
    string toLower(const string& str){
        string copy = str;
        std::transform(copy.begin(), copy.end(), copy.begin(), ::tolower);
        return copy;
    }
    
    void usage(){
        cout << "Use commands: build, run\n";
    }
    
    void build(shared_ptr<DNNGraphSettings>& settings){        
        // Sim command
        settings->command = command;
        
        // Logging level
        settings->logLevel = config["logLevel"].as<int>();
        
        // Write paths
        settings->netPath = config["netPath"].as<std::string>();
        settings->netType = config["netType"].as<std::string>();     
        
        // Topology
        settings->nHLayers = config["topology"]["nHiddenLayers"].as<int>();
        settings->nHidden = config["topology"]["nHidden"].as<int>();
        settings->nInput = config["topology"]["nInput"].as<int>();
        settings->nOutput = config["topology"]["nOutput"].as<int>();
    }
    
    void run(shared_ptr<DNNGraphSettings>& settings){   
        // Sim command
        settings->command = command;
        
        // Logging level
        settings->logLevel = config["logLevel"].as<int>();
        
        // test number
        if(config["testNumber"].IsDefined()) settings->testNumber = config["testNumber"].as<int>();
        
        // Read paths
        settings->netPath = config["netPath"].as<std::string>();
        settings->netType = config["netType"].as<std::string>();
        settings->datasetTrainingPath = config["datasetTrainingPath"].as<std::string>();
        settings->datasetTestingPath = config["datasetTestingPath"].as<std::string>();
        settings->datasetType = config["dataset"].as<std::string>();
        
        // Training parameters
        settings->lr = config["settings"]["lr"].as<float>();
        settings->alpha = config["settings"]["alpha"].as<float>();
        settings->batchSize = config["settings"]["batchSize"].as<int>();
        settings->maxEpoch = config["settings"]["maxEpoch"].as<int>();
        
        // Strategy
        if(config["strategy"].IsDefined())
            settings->dropout = config["strategy"].as<std::string>();
        
        // Parallel Data Parameters
        if(config["settings"]["numModels"].IsDefined()) 
            settings->numModels = config["settings"]["numModels"].as<int>();
        
        // Edge Delay settings
        if(config["edgeSettings"].IsDefined()){
            if(config["edgeSettings"]["enableVariableEdgeDelay"].IsDefined()) 
                settings->enableVariableEdgeDelay = config["edgeSettings"]["enableVariableEdgeDelay"].as<bool>();    
            if(config["edgeSettings"]["distribution"].IsDefined()){
                settings->mean = config["edgeSettings"]["distribution"]["mean"].as<float>();
                settings->std = config["edgeSettings"]["distribution"]["std"].as<float>();
                settings->distribution.param(std::normal_distribution<double>(settings->mean, settings->std).param()); 
                settings->generator.seed(std::random_device{}());
            }
        }
    
        // Async Specific Data Parameters
        if(settings->netType == "async_neural" && config["netSpecificSettings"].IsDefined()){
            settings->forwardDropTolerance = config["netSpecificSettings"]["forwardDropTolerance"].as<float>();
            settings->backwardDropTolerance = config["netSpecificSettings"]["backwardDropTolerance"].as<float>();
            settings->waitTimeFactor = config["netSpecificSettings"]["waitTimeFactor"].as<float>();
            settings->waitTime = settings->waitTimeFactor*settings->std;
        }
        
        // initialise error vectors
        settings->accuracy_validation = Eigen::VectorXf::Zero(settings->maxEpoch,1);
        settings->accuracy_train = Eigen::VectorXf::Zero(settings->maxEpoch,1);
        settings->accuracy_testing = Eigen::VectorXf::Zero(settings->maxEpoch,1);
        settings->error_validation = Eigen::VectorXf::Zero(settings->maxEpoch,1);
        settings->error_training = Eigen::VectorXf::Zero(settings->maxEpoch,1);
        settings->error_testing = Eigen::VectorXf::Zero(settings->maxEpoch,1);
        settings->confusion_matrix_test = Eigen::MatrixXi::Zero(10,10);
    }
};

#endif /* YAML_READER_H */

