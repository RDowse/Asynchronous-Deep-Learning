/* 
 * File:   main.cpp
 * Author: ryan
 *
 * Created on 19 March 2017, 16:39
 */
#include "misc/data_wrapper.h"

#include "tools/loader.h"
#include "tools/dnn_graph.h"

#include "mnist/mnist_reader.hpp"

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
    printf("Starting sim...\n");
    Logging::m_logLevel = 5;
    std::ostream *stats=&std::cout;
    ifstream file;
    file.open(path);
    if(file.is_open()){
        int lineNumber = 0, nNodes = 0, nEdges = 0;
        string type = Loader::readType(lineNumber,file);
        Loader::readHeader(lineNumber,file,nNodes,nEdges);
        Simulator sim(3,nNodes,nEdges,*stats);
        Loader::readBody(lineNumber,file,sim,nNodes,nEdges);
        printf("Loaded graph to sim\n");
        
        printf("Loading data\n");
        //MNISTDataWrapper data("mnist/mnist_train_100.csv","mnist/mnist_test_10.csv");
        XORDataWrapper data("xor_train.csv","xor_test.csv");
        sim.loadInput(&data);
        //sim.run("train");
    } else {
        printf("Unable to open file %s\n",path.c_str());
        return;
    }
    file.close();
}

void buildGraph(string name, int nHidden, int nInput, int nOutput){
    DNNGraph dnngraph(1,nHidden,nInput,nOutput);
    stringstream ss1, ss2;
    ss1 << "w/" << name << ".graph";
    ss2 << "w/" << name << ".dot";
    dnngraph.writeGraph(ss1.str());
    dnngraph.writeGraphviz(ss2.str());
}

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
    //buildGraph("xor",2,2,1);
    //buildGraph("test",2,2,2);
    
    //simulate("w/net.graph");
    simulate("w/xor.graph");

    return 0;
}