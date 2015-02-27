/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * @file RBFLayer.h
 * @class RBFLayer
 * @brief This layer introduces a non-linearity (activation function)
 *
 * @author Clemens-Alexander Brust (ikosa dot de at gmail dot com)
 */

#ifndef CONV_RBFLAYER_H
#define CONV_RBFLAYER_H

#include <random>

#include "CombinedTensor.h"
#include "SimpleLayer.h"

namespace Conv {

class RBFLayer : public SimpleLayer {
public:
  RBFLayer(const unsigned int seed);
  
  // Implementations for SimpleLayer
  bool CreateOutputs (const std::vector< CombinedTensor* >& inputs,
                              std::vector< CombinedTensor* >& outputs);
  bool Connect (const CombinedTensor* input, CombinedTensor* output);
  void FeedForward();
  void BackPropagate();
  void OnLayerConnect (Layer* next_layer);
  
private:
  CombinedTensor* param_ = nullptr;
  std::mt19937 generator_;
};

}

#endif
