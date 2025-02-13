#ifndef MLPACK_METHODS_ANN_DAG_NETWORK_IMPL_HPP
#define MLPACK_METHODS_ANN_DAG_NETWORK_IMPL_HPP

#include "dag_network.hpp"

namespace mlpack {

template<typename InitializationRuleType, 
         typename MatType>
DAGNetwork<
    InitializationRuleType, 
    MatType
>::DAGNetwork(InitializationRuleType initializeRule) :
    initializeRule(initializeRule), 
    layerMemoryIsSet(false),
    inputDimensionsAreSet(false) 
{}

template<typename InitializationRuleType,
         typename MatType>
DAGNetwork<
    InitializationRuleType,
    MatType
>::DAGNetwork(const DAGNetwork& network) :
    initializeRule(network.initializeRule),
    network(network.network),
    parameters(network.parameters),
    inputDimensions(network.inputDimensions),
    predictors(network.predictors),
    responses(network.responses),
    layerMemoryIsSet(false),
    inputDimensionsAreSet(false) 
{}

template<typename InitializationRuleType,
         typename MatType>
DAGNetwork<
    InitializationRuleType,
    MatType
>::DAGNetwork(DAGNetwork&& network) :
    initializeRule(std::move(network.initializeRule)),
    network(std::move(network.network)),
    parameters(std::move(network.parameters)),
    inputDimensions(std::move(network.inputDimensions)),
    predictors(std::move(network.predictors)),
    responses(std::move(network.responses)),
    layerMemoryIsSet(false),
    inputDimensionsAreSet(std::move(network.inputDimensionsAreSetl)) 
{}

template<typename InitializationRuleType,
         typename MatType>
DAGNetwork<InitializationRuleType, MatType>& DAGNetwork<
    InitializationRuleType,
    MatType
>::operator=(const DAGNetwork& other)
{
  if (this != &other)
  {
    initializeRule = other.initializeRule;
    network = other.network;
    parameters = other.parameters;
    inputDimensions = other.inputDimensions;
    predictors = other.predictors;
    responses = other.responses;
    inputDimensionsAreSet = other.inputDimensionsAreSet;
    layerMemoryIsSet = false;
  }

  return *this;
}

template<typename InitializationRuleType,
         typename MatType>
DAGNetwork<InitializationRuleType, MatType>& DAGNetwork<
    InitializationRuleType,
    MatType
>::operator=(DAGNetwork&& other)
{
  if (this != &other)
  {
    initializeRule = std::move(other.initializeRule);
    network = std::move(other.network);
    parameters = std::move(other.parameters);
    inputDimensions = std::move(other.inputDimensions);
    predictors = std::move(other.predictors);
    responses = std::move(other.responses);
    inputDimensionsAreSet = std::move(other.inputDimensionsAreSet);
    layerMemoryIsSet = std::move(other.layerMemoryIsSet);
  }

  return *this;
}

} // namespace mlpack

#endif
