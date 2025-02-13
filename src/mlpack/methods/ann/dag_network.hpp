#ifndef MLPACK_METHODS_ANN_DAG_NETWORK_HPP
#define MLPACK_METHODS_ANN_DAG_NETWORK_HPP

#include "init_rules/init_rules.hpp"

namespace mlpack {

template<
    typename InitializationRuleType = RandomInitialization,
    typename MatType = arma::mat>
class DAGNetwork 
{
public:
  DAGNetwork(InitializationRuleType initializeRule = InitializationRuleType());

  DAGNetwork(const DAGNetwork& other);
  DAGNetwork(DAGNetwork&& other);
  DAGNetwork& operator=(const DAGNetwork& other);
  DAGNetwork& operator=(DAGNetwork&& other);

  void Link(Layer<MatType>* layerA, Layer<MatType>* layerB);

  // Get the network, sorted topologically
  const std::vector<Layer<MatType>*>& Network() const;

  void Predict(const MatType& predictors,
               MatType& results,
               const size_t batchSize = 128);

  size_t WeightSize();

  // Get / set the input dimensions
  const std::vector<size_t>& InputDimensions() const;
  std::vector<size_t>& InputDimensions();

  // Get / set the parameters
  const MatType& Parameters() const;
  MatType& Parameters();

  void Reset();

  void SetNetworkMode(const bool training);

  void Forward(const MatType& inputs, MatType& results);

private:

  void InitializeWeights();
  void SetLayerMemory();

  void CheckGraph(const std::string& functionName,
                  const bool setMode = false,
                  const bool training = false);

  void UpdateDimensions(const std::string& functionName);

  InitializationRuleType initializeRule;

  std::vector<Layer<MatType>*> network;

  MatType parameters;

  std::vector<size_t> inputDimensions;

  MatType predictors;
  MatType responses;

  bool layerMemoryIsSet;
  bool inputDimensionsAreSet;

};

} // namespace mlpack

#include "dag_network_impl.hpp"

#endif
