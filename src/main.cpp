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
#include <ctime>
#include <random>
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
        Simulator sim(1,nNodes,nEdges,*stats);
        Loader::readBody(lineNumber,file,sim,nNodes,nEdges);
        printf("Loaded graph to sim\n");
        
        printf("Loading data\n");
//        auto dataset = mnist::read_dataset<std::vector,std::vector, uint8_t, uint8_t>();
//        sim.loadInput(&dataset);
        DataWrapper data("mnist/mnist_train_100.csv","mnist/mnist_test_10.csv");
        sim.loadInput(&data);
        sim.run("train");
    } else {
        printf("Unable to open file %s\n",path.c_str());
        return;
    }
    file.close();
}

//void simulateTest(const string& path){
//    srand( time(NULL) );
//    printf("Staring sim...\n");
//    Logging::m_logLevel = 5;
//    std::ostream *stats=&std::cout;
//    ifstream file;
//    file.open(path);
//    if(file.is_open()){
//        int lineNumber = 0, nNodes = 0, nEdges = 0;
//        string type = Loader::readType(lineNumber,file);
//        Loader::readHeader(lineNumber,file,nNodes,nEdges);
//        Simulator sim(3,nNodes,nEdges,*stats);
//        Loader::readBody(lineNumber,file,sim,nNodes,nEdges);
//        printf("Loaded graph to sim\n");
//        
//        Loader::loadWeights("tut_w.csv", sim);      
//        vector<float> d = {0.05,0.1};
//        //sim.loadInput(d);
//        sim.run("predict");
//        sim.printOutput();
//        
//        sim.run("predict");
//        sim.printOutput();
//        
//        // End sim
//    } else {
//        printf("Unable to open file %s\n",path.c_str());
//        return;
//    }
//}

void buildGraph(string name, int nHidden, int nInput, int nOutput){
    DNNGraph dnngraph(1,nHidden,nInput,nOutput);
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
    
    //buildGraph("net",10,28*28,10);
    //buildGraph("test",2,2,2);
    
    simulate("w/net.graph");

    return 0;
}