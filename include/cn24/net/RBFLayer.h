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

#include "CombinedTensor.h"
#include "SimpleLayer.h"

namespace Conv {

class RBFLayer : public SimpleLayer {
public:
  RBFLayer();
  
  // Implementations for SimpleLayer
  bool CreateOutputs (const std::vector< CombinedTensor* >& inputs,
                              std::vector< CombinedTensor* >& outputs);
  bool Connect (const CombinedTensor* input, CombinedTensor* output);
  void FeedForward();
  void BackPropagate();

};

}

#endif
