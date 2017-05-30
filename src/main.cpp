/* 
 * File:   main.cpp
 * Author: ryan
 *
 * Created on 19 March 2017, 16:39
 */
#include "misc/yaml_reader.h"

#include "misc/data_wrapper.h"

#include "nodes/neural_node.h"
#include "nodes/pardata_nodes/parallel_data_neural_node.h"

#include "tools/loader.h"
#include "tools/dnn_graph.h"

#include "misc/node_factory.h"

#include <map>
#include <vector>
#include <iostream>
#include <memory>
#include <ctime>
#include <random>
#include <cstdlib>

void usage(){
    cout << "Usage: enter yaml path for configuration\n";
}

template<typename TNode>
void simulate(shared_ptr<DNNGraphSettings> settings){
    printf("Starting sim...\n");
    Logging::m_logLevel = 5;
    std::ostream *stats=&std::cout;
    ifstream file;
    file.open(settings->netPath);
    if(file.is_open()){
        int lineNumber = 0, nNodes = 0, nEdges = 0;
        string type = Loader::readType(lineNumber,file);
        
        Loader::readHeader(lineNumber,file,settings,nNodes,nEdges);
        
        cout << settings->logLevel << endl;
        Simulator<TNode> sim(settings->logLevel,nNodes,nEdges,*stats);
        Loader::readBody(lineNumber,file,settings,sim,nNodes,nEdges);
        printf("Loaded graph to sim\n");
        
        vector<string> conf = {settings->dropout};
        sim.setStrategies(conf);
        
        printf("Loading data\n");
        DataWrapper* data;
        if("mnist" == settings->datasetType){
            data = new MNISTDataWrapper(settings->datasetTrainingPath,settings->datasetTestingPath);
        } else if("xor" == settings->datasetType){
            data = new XORDataWrapper(settings->datasetTrainingPath,settings->datasetTestingPath);
        } else {
            cout << "Invalid dataset type\n" << endl;
            assert(0);
        }
        
        // basic check for data size
        assert(data->training_images.cols() < nNodes);
        sim.loadInput(data);
        cout << "Starting run\n";
        sim.run("train");
        
        auto s = std::static_pointer_cast<DNNGraphSettings>(settings);
        cout << "Final epoch " << s->epoch << endl;
        cout << "training error: " << s->training_error << ", accuracy: " << s->accuracy << endl;
    } else {
        cout << "Unable to open file "<< settings->netPath << "\n";
        return;
    }
    file.close();
}

template<typename TNode>
void buildGraph(string name, int nHLayers, int nHidden, int nInput, int nOutput, bool bias, int nCPU=4){
    DNNGraphBuilder<TNode> dnngraph(nHLayers,nHidden,nInput,nOutput,bias,nCPU);
    stringstream ss1, ss2;
    ss1 << name << ".graph";
    ss2 << name << ".dot";
    dnngraph.writeGraph(ss1.str());
    dnngraph.writeGraphviz(ss2.str());
}

template<typename TNode>
void run(shared_ptr<DNNGraphSettings> settings){
    registerNodes<TNode>();
    if(settings->command == "build"){
        buildGraph<TNode>(settings->netPath, settings->nHLayers, settings->nHidden,
                settings->nInput, settings->nOutput, true);
    } else if(settings->command == "run") {
        simulate<TNode>(settings);
    } else {
        assert(0);
    }
}

int main(int argc, char** argv) {
    // Get usage for the executable.
    if(argc!=2){
        usage();
        return 0;
    }
    srand(time(NULL));
    
    string yamlPath = argv[1];
    YamlReader yamlReader(yamlPath);
    auto settings = make_shared<DNNGraphSettings>();
    yamlReader.readConfig(settings);
    
    settings->printParameters();
    
    if("neural" == settings->netType){
        run<NeuralNode>(settings);
    } else if("parallel_data_neural" == settings->netType){
        run<ParallelDataNeuralNode>(settings);
    } else {
        cout << "Net type " << settings->netType << "not recognised" << endl;
    }

    return 0;
}

