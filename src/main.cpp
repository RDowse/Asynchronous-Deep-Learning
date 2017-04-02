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

#include "tools/loader.h"
#include "tools/dnn_graph.h"

#include "mnist/mnist_reader.hpp"

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
    
//    DNNGraph dnngraph(1,4,5,5);
//    dnngraph.printGraph("w/test.graph");
//    dnngraph.printGraphviz("w/test.dot");
//    
//    Simulator sim(2,*stats);
//    Loader::load("w/test.graph",sim);
//    sim.run("predict");

    //Data<int,vector<int>> data;
    //DataLoader::load(data,"data/mnist_train.csv");
    
//    vector<vector<double>> v;
//    load_mnist::ReadMNIST(10000,784,v);
//    for(auto i: v[0]){
//        cout << i << " ";
//    }
    
    auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();
    for(int i = 0; i<dataset.training_labels.size(); ++i){   
        cout << int(dataset.training_labels[i]) << " ";
    }
//    for(int i = 0; i<dataset.training_images[0].size(); ++i){
//        cout << int(dataset.training_images[0][i]) << " ";
//    }
    return 0;
}

