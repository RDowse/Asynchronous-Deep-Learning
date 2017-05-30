
#include "tools/dnn_graph.h"

template<> DNNGraphBuilder<NeuralNode>::DNNGraphBuilder(int _nHLayers, int _nHidden, 
        int _nInput, int _nOutput, bool _bias, int _nCPU)
    :nHLayers(_nHLayers),nHidden(_nHidden),nInput(_nInput),nOutput(_nOutput),bias(_bias),
        actNHidden(_nHidden), actNInput(_nInput), actNOutput(_nOutput)
{
    init();
}

template<> string DNNGraphBuilder<NeuralNode>::type(){
    return "NeuralNode";
}

template<> void DNNGraphBuilder<NeuralNode>::writeParams(ofstream& file){
    file<<4<<"\n";        // PARAMETER COUNT
    file<<nHLayers<<" "<<nHidden<<" "<<nInput<<" "<<nOutput<<"\n";
}

template<> DNNGraphBuilder<ParallelDataNeuralNode>::DNNGraphBuilder(int _nHLayers, int _nHidden, 
        int _nInput, int _nOutput, bool _bias, int _nCPU)
    :nHLayers(_nHLayers),nHidden(_nHidden),nInput(_nInput),nOutput(_nOutput),bias(_bias),
        actNHidden(_nHidden), actNInput(_nInput), actNOutput(_nOutput)
{
    init();
}

template<> string DNNGraphBuilder<ParallelDataNeuralNode>::type(){
    return "ParallelDataNeuralNode";
}

template<> void DNNGraphBuilder<ParallelDataNeuralNode>::writeParams(ofstream& file){
    file<<4<<"\n";        // PARAMETER COUNT
    file<<nHLayers<<" "<<nHidden<<" "<<nInput<<" "<<nOutput<<"\n";
}

//template<> DNNGraphBuilder<BlockNeuralNode>::DNNGraphBuilder(int _nHLayers, int _nHidden, 
//        int _nInput, int _nOutput, bool _bias, int _nCPU)
//    :nHLayers(_nHLayers),nHidden(_nCPU),nInput(_nCPU),nOutput(_nCPU),bias(_bias),
//        actNHidden(_nHidden), actNInput(_nInput), actNOutput(_nOutput)
//{
//    init();
//}
//
//template<> string DNNGraphBuilder<BlockNeuralNode>::type(){
//    return "BlockNeuralNode";
//}
//
//template<> void DNNGraphBuilder<BlockNeuralNode>::writeParams(ofstream& file){
//    file<<7<<"\n";        // PARAMETER COUNT
//    file<<nHLayers<<" "<<actNHidden<<" "<<actNInput<<" "<<actNOutput<<" "; 
//    file<<nHidden<<" "<<nInput<<" "<<nOutput<<"\n";
//}