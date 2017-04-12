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
        Simulator sim(0,nNodes,nEdges,*stats);
        Loader::readBody(lineNumber,file,sim,nNodes,nEdges);
        printf("Loaded graph to sim\n");
        
        // Sim
        sim.setup();
        
//        printf("Loading data\n");
//        MNISTDatasetWrapper data(mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>());
//        printf("Data size: %ld\n",data.getData()[0].size());
//        
//        vector<int> removedIndex; // remove redundant columns of pixels.
//        math::removeConstantCols(data.getData(),removedIndex);
//        
//        Loader::loadWeights("weights.csv", sim, removedIndex);
//        int label = 2;
//        sim.loadInput(data.getData()[label]);
//        sim.run("predict");
//        /*
//         Weights are possibly not in the order i think they are.
//         */
//        sim.printOutput();
//        cout << "LABEL: " << int(data.getLabels()[label]) << endl;
        
//        Loader::loadWeights("tut_w.csv", sim);      
//        vector<float> d = {0.05,0.1};
//        sim.loadInput(d);
//        sim.run("predict");
//        sim.printOutput();
        
        Loader::loadWeights("data/net/w.csv", sim);
        vector<float> d = {0.12304,-0.70788};
        //vector<float> d = {0.30233,-5.9132};
        //vector<float> d = {0.16425,-2.8781};
        
        sim.loadInput(d);
        sim.run("predict");
        sim.printOutput();
        cout<<"INPUT "<<d[0]<<" "<<d[1]<<endl;
        
        // End sim
    } else {
        printf("Unable to open file %s\n",path.c_str());
        return;
    }
    file.close();
}

void buildGraph(string name){
    //DNNGraph dnngraph(1,10,28*28,10);
    DNNGraph dnngraph(1,2,2,2);
    stringstream ss1, ss2;
    ss1 << "w/" << name << ".graph";
    ss2 << "w/" << name << ".dot";
    dnngraph.writeGraph(ss1.str());
    dnngraph.writeGraphviz(ss2.str());
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

    std::ostream *stats=&std::cout;
    
    string name = "tut";
    //buildGraph(name);
    //std::this_thread::sleep_for (std::chrono::seconds(1));
    //simulate("w/test.graph");
    simulate("w/tut.graph");

    return 0;
}