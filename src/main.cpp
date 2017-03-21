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
#include "misc/node_factory.h"
#include "graphs/basic_graph.h"
#include "graphs/dnn_graph.h"
#include "nodes/basic_node.h"

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

    auto basic_graph = make_shared<BasicGraph>();
    auto dnn_graph = make_shared<DNNGraph>();
    auto node = NodeFactory::createInstance("Basic",dnn_graph);
    
    return 0;
}

