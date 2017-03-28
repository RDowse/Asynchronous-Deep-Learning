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

#include <algorithm>
#include <vector>
#include <memory>
#include <iostream>
#include <fstream>
#include <stdio.h>

using namespace std;

class DNNGraph{
    struct Edge{
        int src;
        int dst;
        int delay;
        Edge(int src, int dst, int delay):src(src),dst(dst),delay(delay){}
    };
    vector<Node*> nodes;
    vector<Edge*> edges;
    int nHLayers=0, nHidden=0, nInput=0, nOutput=0;
    int clusterCount = 0;
public:    
    DNNGraph(int nHLayers, int nHidden, int nInput, int nOutput)
        :nHLayers(nHLayers),nHidden(nHidden),nInput(nInput),nOutput(nOutput)
    {
        auto settings = make_shared<DNNGraphSettings>();
        
        vector<Node*> prev_layer;
        vector<Node*> curr_layer;
        
        // Input nodes
        for(int i = 0; i < nInput; ++i){
            prev_layer.push_back(new InputNode(settings));
        }
        
        // Hidden nodes
        for(int i = 0; i < nHLayers; ++i){
            // Layer
            for(int j = 0; j < nHidden; ++j){
                curr_layer.push_back(new DNNNode(settings));    
            }
            // Connect Edges
            for(int j = 0; j < prev_layer.size(); ++j){
                for(int k = 0; k < curr_layer.size(); ++k){
                    edges.push_back(
                        new Edge(prev_layer[j]->getId(),curr_layer[k]->getId(),1)
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
            curr_layer.push_back(new OutputNode(settings));
        }
        // Connect Edges
        for(int j = 0; j < prev_layer.size(); ++j){
            for(int k = 0; k < curr_layer.size(); ++k){
                edges.push_back(
                    new Edge(prev_layer[j]->getId(),curr_layer[k]->getId(),1)
                );
            }
        }
        
        // Copy node layers
        nodes.insert(nodes.end(),prev_layer.begin(),prev_layer.end());
        nodes.insert(nodes.end(),curr_layer.begin(),curr_layer.end());
    }
    
    ~DNNGraph(){
        for (auto& i: nodes){
          delete (i);
        } 
        nodes.clear();
        for (auto& i: edges){
          delete (i);
        } 
        edges.clear();
    }
    
    void printGraph(const string& path){
        ofstream file;
        file.open(path);
        if(file.is_open()){
            printGraphHeader(file);
            printGraphNodes(file);
            printGraphEdges(file);
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
            fprintf(file,"node [fixedsize=true, label=""];\n");
            // Input nodes
            printCluster(file,index,index+nInput,nodes[index]->getType(),"blue4");
            index = nInput;
            // Hidden nodes
            for(int i = 0; i < nHLayers; ++i)
                printCluster(file,index+i*nHidden,index+(i+1)*nHidden,nodes[index]->getType(),"red2");
            index += nHidden*nHLayers;
            // Output nodes
            printCluster(file,index,index+nOutput,nodes[index]->getType(),"seagreen2");
            fprintf(file,"\n");
            fprintf(file,"}");
        } else {
            std::cout<< "Error opening file: %s" << path;
        }
        fclose(file);
    }
 
private:
    void printGraphHeader(ofstream& file){
        file<<"POETSGraph\nBeginHeader\n";
        // TYPE
        file<<"DNN"<<"\n";
        // PARAMETER COUNT
        file<<"0"<<"\n";
        // PARAMETERS
        file<<"NULL"<<"\n";
        // NODES AND EDGES COUNT
        file<<nodes.size()<<" "<<edges.size()<<"\n";
        file<<"EndHeader\n";            
    }
    
    void printGraphNodes(ofstream& file){
        file<<"BeginNodes\n";
        for(auto n: nodes){
            file<<n->getType()<<" "<<n->getId()<<"\n";
        }
        file<<"EndNodes\n";
    }
    
    void printGraphEdges(ofstream& file){
        file<<"BeginEdges\n";
        for(auto e: edges){
            file<<" "<<e->src<<" "<<e->dst<<" "<<e->delay<<"\n";
        }
        file<<"EndNodes\n";
    }
    
    void printCluster(FILE* file, int start, int end, string prefix, string color){
        fprintf(file,"subgraph cluster_%d{\n",clusterCount++);
        fprintf(file,"    color=%s\n","white");
        fprintf(file,"    node [style=solid,color=%s,shape=circle];\n    ",color.c_str());
        for(int i = start; i < end; ++i)
            fprintf(file,"%s%d ",prefix.c_str(),nodes[i]->getId());
        fprintf(file,";\n");
        fprintf(file,"    label = \"%s layer\";\n}\n",prefix.c_str());
    }
    
    void printConnections(FILE* file){
        for(auto e: edges){
            //fprintf(file,
        }
    }
};

#endif /* DNN_GRAPH_H */

