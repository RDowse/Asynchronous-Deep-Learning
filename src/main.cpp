/* 
 * File:   main.cpp
 * Author: ryan
 *
 * Created on 19 March 2017, 16:39
 */
#include "misc/yaml_reader.h"

#include "misc/data_wrapper.h"

#include "nodes/neural_node.h"
#include "nodes/async_nodes/async_neural_node.h"
#include "nodes/pardata_nodes/parallel_data_neural_node.h"

#include "tools/loader.h"
#include "tools/dnn_graph.h"
#include "tools/csv_writer.h"

#include "misc/node_factory.h"

#include "tbb/tick_count.h"

#include <map>
#include <vector>
#include <iostream>
#include <memory>
#include <ctime>
#include <random>
#include <cstdlib>

string testPath = "";

void usage(){
    cout << "Usage: enter yaml path for configuration\n";
}

template<typename TNode>
void simulate(shared_ptr<DNNGraphSettings> context){
    printf("Starting sim...\n\n");
    Logging::m_logLevel = 5;
    std::ostream *stats=&std::cout;
    ifstream file;
    file.open(context->netPath);
    if(file.is_open()){
        int lineNumber = 0, nNodes = 0, nEdges = 0;
        string type = Loader::readType(lineNumber,file);
        
        Loader::readHeader(lineNumber,file,context,nNodes,nEdges);
        
        cout << context->logLevel << endl;
        Simulator<TNode> sim(context->logLevel,nNodes,nEdges,*stats);
        Loader::readBody(lineNumber,file,context,sim,nNodes,nEdges);
        printf("Loaded graph to sim\n");
        
        vector<string> conf = {context->dropout};
        sim.setStrategies(conf);
        
        printf("Loading data\n");
        DataWrapper* data;
        
        if("mnist" == context->datasetType){
            data = new MNISTDataWrapper(context->datasetTrainingPath,context->datasetTestingPath);
        } else if("xor" == context->datasetType){
            data = new XORDataWrapper(context->datasetTrainingPath,context->datasetTestingPath);
        } else {
            cout << "Invalid dataset type\n" << endl;
            assert(0);
        }
        
        // basic check for data size
        assert(data->training_images.cols() < nNodes);
        sim.loadInput(data);
        
        // run
        cout << "Starting run\n";
        sim.run("train");
        
        auto s = std::static_pointer_cast<DNNGraphSettings>(context);
        cout << "Final epoch: " << s->epoch << endl;
        cout << "training error: " << s->training_error << ", accuracy: " << s->accuracy << endl;
        
        if(testPath!=""){
            stringstream ss;
            ss << testPath << "/";
            CSVWriter::writeCSV<Eigen::VectorXf,Eigen::MatrixXf>(
                ss.str()+"error.csv",context->error_training,context->error_validation,context->error_testing,
                "training error, validation error, testing error \n");
            CSVWriter::writeCSV<Eigen::VectorXf,Eigen::MatrixXf>(
                ss.str()+"accuracy.csv",context->accuracy_train,context->accuracy_validation,context->accuracy_testing,
                "training accuracy, validation accuracy, testing accuracy \n");
            CSVWriter::writeContext(ss.str()+"config.txt",context);
        }
        
        delete data;
    } else {
        cout << "Unable to open file "<< context->netPath << "\n";
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
void run(shared_ptr<DNNGraphSettings> context){
    registerNodes<TNode>();
    if(context->command == "build"){
        buildGraph<TNode>(context->netPath, context->nHLayers, context->nHidden,
                context->nInput, context->nOutput, true);
    } else if(context->command == "run") {
        simulate<TNode>(context);
    } else {
        assert(0);
    }
}

int main(int argc, char** argv) {
    // Get usage for the executable.
    if(argc<2 || argc>3){
        usage();
        return 0;
    }
    srand(time(NULL));
    
    if(argc==3){
        testPath = argv[2];
    }
    
    string yamlPath = argv[1];
    YamlReader yamlReader(yamlPath);
    auto context = make_shared<DNNGraphSettings>();
    yamlReader.readConfig(context);
    
    context->printParameters();
    
    if("neural" == context->netType){
        run<NeuralNode>(context);
    } else if("parallel_data_neural" == context->netType){
        run<ParallelDataNeuralNode>(context);
    } else if("async_neural" == context->netType){
        run<AsyncNeuralNode>(context);
    }else {
        cout << "Net type " << context->netType << "not recognised" << endl;
    }

    return 0;
}

