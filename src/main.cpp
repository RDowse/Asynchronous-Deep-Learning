/* 
 * File:   main.cpp
 * Author: ryan
 *
 * Created on 19 March 2017, 16:39
 */
#include "misc/data_wrapper.h"

#include "nodes/neural_node.h"
#include "nodes/block_nodes/block_neural_node.h"

#include "tools/loader.h"
#include "tools/dnn_graph.h"

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

void printData(const DataWrapper& data){
    cout << "Images: " << endl;
    cout << data.training_images << "\n";
    cout << "Labels: " << endl;
    cout << data.training_labels << "\n";
}

template<typename TNode>
void simulate(const string& path){
    printf("Starting sim...\n");
    Logging::m_logLevel = 5;
    std::ostream *stats=&std::cout;
    ifstream file;
    file.open(path);
    if(file.is_open()){
        int lineNumber = 0, nNodes = 0, nEdges = 0;
        string type = Loader::readType(lineNumber,file);
        shared_ptr<GraphSettings> settings;
        
        if(type == "BlockNeuralNode"){
            settings = make_shared<BlockNeuralNetworkSettings>();
        } else if(type == "NeuralNode"){
            settings = make_shared<DNNGraphSettings>();
        } else {
            cout << "Invalid graph type\n";
            assert(0);
        }
        Loader::readHeader(lineNumber,file,settings,nNodes,nEdges);
        
        Simulator<TNode> sim(0,nNodes,nEdges,*stats);
        Loader::readBody(lineNumber,file,settings,sim,nNodes,nEdges);
        printf("Loaded graph to sim\n");
        
        printf("Loading data\n");
        MNISTDataWrapper data("mnist/mnist_train_100.csv","mnist/mnist_test_10.csv");
        //XORDataWrapper data("xor_train.csv","xor_train.csv");
        sim.loadInput(&data);
        sim.run("train");
        
        cout << "Final epoch " << std::static_pointer_cast<DNNGraphSettings>(settings)->epoch << "\n";
    } else {
        printf("Unable to open file %s\n",path.c_str());
        return;
    }
    file.close();
}

template<typename T>
void buildGraph(string name, int nHLayers, int nHidden, int nInput, int nOutput, bool bias, int nCPU=4){
    DNNGraphBuilder<T> dnngraph(1,nHidden,nInput,nOutput,bias,nCPU);
    stringstream ss1, ss2;
    ss1 << "w/" << name << ".graph";
    ss2 << "w/" << name << ".dot";
    dnngraph.writeGraph(ss1.str());
    dnngraph.writeGraphviz(ss2.str());
}

int main(int argc, char** argv) {
    // Get usage for the executable.
//    if(argc==1){
//        //usage();
//        return 0;
//    }
//    
//    // Map of flag functions
//    std::map<std::string, FlagFunction> func_map;
//    registerFlags(func_map);
//
//    // choose operation type based on flags, eg. -s save
//    auto func = func_map.find(argv[1]);
//    if(func!= func_map.end()){
//        std::vector<char*> param(argv+1,argv+argc);
//        func_map[argv[1]](param);
//    } else {
//        printf("Flag %s does not exist.", argv[1]);
//        exit(1);
//    }
    
    std::ostream *stats=&std::cout;
    //buildGraph<NeuralNode>("net",1,10,28*28,10,true);
    //buildGraph<NeuralNode>("net2",2,10,28*28,10,true);
    //buildGraph<NeuralNode>("xor",1,2,2,1,true); 
    //buildGraph<NeuralNode>("test2",2,3,3,2,true); 
    
    //simulate<NeuralNode>("w/net.graph");
    simulate<NeuralNode>("w/net2.graph");
    //simulate<NeuralNode>("w/xor.graph");
    
    return 0;
}