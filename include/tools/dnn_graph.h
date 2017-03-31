/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   dnn_graph.h
 * Author: ryan
 *
 * Created on 23 March 2017, 23:26
 */

#ifndef DNN_GRAPH_H
#define DNN_GRAPH_H

#include "graphs/dnn_graph_settings.h"

#include "nodes/node.h"
#include "nodes/dnn_node.h"
#include "nodes/input_node.h"
#include "nodes/output_node.h"

#include "misc/edge.h"

#include <algorithm>
#include <vector>
#include <memory>
#include <iostream>
#include <fstream>
#include <stdio.h>

using namespace std;

class DNNGraph{
    vector<shared_ptr<Node>> nodes;
    vector<shared_ptr<Edge>> edges;
    int nHLayers=0, nHidden=0, nInput=0, nOutput=0;
    int clusterCount = 0;
    float thickness = 0.2;
public:    
    DNNGraph(int nHLayers, int nHidden, int nInput, int nOutput)
        :nHLayers(nHLayers),nHidden(nHidden),nInput(nInput),nOutput(nOutput)
    {
        auto settings = make_shared<DNNGraphSettings>();
        
        vector<shared_ptr<Node>> prev_layer;
        vector<shared_ptr<Node>> curr_layer;
        // Input nodes
        for(int i = 0; i < nInput; ++i){
            prev_layer.push_back(make_shared<InputNode>(settings));
        }
        
        // Hidden nodes
        for(int i = 0; i < nHLayers; ++i){
            // Layer
            for(int j = 0; j < nHidden; ++j){
                curr_layer.push_back(make_shared<DNNNode>(settings));    
            }
            // Connect Edges
            for(int j = 0; j < prev_layer.size(); ++j){
                for(int k = 0; k < curr_layer.size(); ++k){
                    edges.push_back(
                        make_shared<Edge>(prev_layer[j],curr_layer[k],1)
                    );
                    edges.push_back(
                        make_shared<Edge>(curr_layer[k],prev_layer[j],1)
                    );
                }
            }
            // Copy node layers
            nodes.insert(nodes.end(),prev_layer.begin(),prev_layer.end());
            swap(curr_layer,prev_layer);
            curr_layer.clear();
        }
        
        // Output nodes
        for(int i = 0; i < nOutput; ++i){
            curr_layer.push_back(make_shared<OutputNode>(settings));
        }
        // Connect Edges
        for(int j = 0; j < prev_layer.size(); ++j){
            for(int k = 0; k < curr_layer.size(); ++k){
                edges.push_back(
                    make_shared<Edge>(prev_layer[j],curr_layer[k],1)
                );
                edges.push_back(
                    make_shared<Edge>(curr_layer[k],prev_layer[j],1)
                );
            }
        }
        
        // Copy node layers
        nodes.insert(nodes.end(),prev_layer.begin(),prev_layer.end());
        nodes.insert(nodes.end(),curr_layer.begin(),curr_layer.end());
    }
    
    ~DNNGraph(){}
    
    void printGraph(const string& path){
        ofstream file;
        file.open(path);
        if(file.is_open()){
            file<<"POETSGraph\n";
            file<<"DNN"<<"\n";      // TYPE
            
            file<<"BeginHeader\n";
            file<<"0"<<"\n";        // PARAMETER COUNT
            file<<"NULL"<<"\n";     // PARAMETERS
            file<<nodes.size()<<" "<<edges.size()<<"\n";
            file<<"EndHeader\n";   
            
            file<<"BeginNodes\n";
            for(auto n: nodes){
                file<<n->getType()<<" "<<n->getId()<<"\n";
            }
            file<<"EndNodes\n";
            
            file<<"BeginEdges\n";
            for(auto e: edges){
            file<<e->src->getId()<<" "<<e->dst->getId()<<" "<<e->getDelay()<<"\n";
            }
            file<<"EndEdges\n";
        } else {
            std::cout<< "Error opening file: %s" << path;
        }
        file.close();
    }
 
    void printGraphviz(const string& path){
        FILE* file;
        file = fopen(path.c_str(),"w");
        if(file){
            int index = 0;
            fprintf(file,"digraph G {\n");
            fprintf(file,"rankdir=LR\n");
            fprintf(file,"splines=line\n");
            fprintf(file,"node [fixedsize=true, label=\"\"];\n");
            
            // Input nodes
            printGraphvizCluster(file,index,
                    index+nInput,nodes[index]->getType(),"blue4");
            index = nInput;
            
            // Hidden nodes
            for(int i = 0; i < nHLayers; ++i)
                printGraphvizCluster(file,index+i*nHidden,
                        index+(i+1)*nHidden,nodes[index]->getType(),"red2");
            index += nHidden*nHLayers;
            
            // Output nodes
            printGraphvizCluster(file,index,
                    index+nOutput,nodes[index]->getType(),"seagreen2");
            fprintf(file,"\n");
            printGraphvizConnections(file);
            fprintf(file,"}");
        } else {
            std::cout<< "Error opening file: %s" << path;
        }
        fclose(file);
    }
 
private:
    void printGraphvizCluster(FILE* file, int start, int end, string prefix, string color){
        fprintf(file,"subgraph cluster_%d{\n",clusterCount++);
        fprintf(file,"    color=%s\n","white");
        fprintf(file,"    node [style=solid,color=%s,shape=circle];\n    ",color.c_str());
        for(int i = start; i < end; ++i)
            fprintf(file,"%s%d ",prefix.c_str(),nodes[i]->getId());
        fprintf(file,";\n");
        fprintf(file,"    label = \"%s layer\";\n}\n",prefix.c_str());
    }
    
    void printGraphvizConnections(FILE* file){
        for(auto e: edges){
            fprintf(file,"%s%d -> %s%d [penwidth=%f]\n",e->src->getType().c_str(),e->src->getId(),
                            e->dst->getType().c_str(),e->dst->getId(),thickness);
        }
    }
};

#endif /* DNN_GRAPH_H */

