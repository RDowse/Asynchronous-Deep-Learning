/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   main.cpp
 * Author: ryan
 *
 * Created on 19 March 2017, 16:39
 */
#include "graphs/basic_graph_settings.h"
#include "graphs/dnn_graph_settings.h"
#include "nodes/basic_node.h"
#include "messages/message.h"
#include "messages/forward_propagation_message.h"

#include "misc/node_factory.h"
#include "misc/data.h"
#include "misc/data_wrapper.h"

#include "tools/loader.h"
#include "tools/dnn_graph.h"

#include "mnist/mnist_reader.hpp"
#include "tools/math.h"

#include <map>
#include <vector>
#include <iostream>
#include <memory>
#include <cstdlib>

typedef void (*FlagFunction)(const std::vector<char*>& parameters);

void save(const std::vector<char*>& parameters){
    std::cout << "Save\n";
}

void run(const std::vector<char*>& parameters){
    std::cout << "Run\n";
}

void usage(){
    std::cout << "Usage:\n";
    std::cout << "flags -s -l -r\n";
}

void registerFlags(std::map<std::string, FlagFunction>& func_map){
    func_map["-s"] = &save;
    func_map["-r"] = &run;
}

void simulate(const string& path){
    printf("Staring sim...\n");
    Logging::m_logLevel = 5;
    std::ostream *stats=&std::cout;
    ifstream file;
    file.open(path);
    if(file.is_open()){
        int lineNumber = 0, nNodes = 0, nEdges = 0;
        string type = Loader::readType(lineNumber,file);
        Loader::readHeader(lineNumber,file,nNodes,nEdges);
        Simulator sim(2,nNodes,nEdges,*stats);
        Loader::readBody(lineNumber,file,sim,nNodes,nEdges);
        printf("Loaded graph to sim\n");
        
        // Sim
        printf("Loading data\n");
        MNISTDatasetWrapper data(mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>());
        printf("Data size: %ld\n",data.getData()[0].size());
        
        vector<int> removedIndex; // remove redundant columns of pixels.
        math::removeConstantCols(data.getData(),removedIndex);
        
        vector<float> weights;
        Loader::loadWeights("weights.csv", weights);
        
        for(auto w: weights){
            cout << w << endl;
        }
        
        sim.loadInput(data.getData()[0]);
        // End sim
    } else {
        printf("Unable to open file %s\n",path.c_str());
        return;
    }
    file.close();
}

void buildGraph(){
    DNNGraph dnngraph(1,10,28*28,10);
    dnngraph.writeGraph("w/test.graph");
    dnngraph.writeGraphviz("w/test.dot");
}

/*
 * 
 */
int main(int argc, char** argv) {
    // Get usage for the executable.
    if(argc==1){
        usage();
        return 0;
    }
    
    // Map of flag functions
    std::map<std::string, FlagFunction> func_map;
    registerFlags(func_map);

    // choose operation type based on flags, eg. -s save
    auto func = func_map.find(argv[1]);
    if(func!= func_map.end()){
        std::vector<char*> param(argv+1,argv+argc);
        func_map[argv[1]](param);
    } else {
        printf("Flag %s does not exist.", argv[1]);
        exit(1);
    }

    Logging::m_logLevel = 5;
    std::ostream *stats=&std::cout;
    
    //buildGraph();
    simulate("w/test.graph");

    return 0;
}