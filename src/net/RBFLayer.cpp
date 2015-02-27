/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
#include <vector>
#include <math.h>

#include "CombinedTensor.h"
#include "Log.h"

#include "RBFLayer.h"

namespace Conv {
  
RBFLayer::RBFLayer(const unsigned int seed) : generator_(seed) {
}


bool RBFLayer::CreateOutputs (
  const std::vector< CombinedTensor* >& inputs,
  std::vector< CombinedTensor* >& outputs) {
  // This is a simple layer, only one input
  if (inputs.size() != 1) {
    LOGERROR << "Only one input supported!";
    return false;
  }

  // Save input node pointer
  CombinedTensor* input = inputs[0];

  // Check if input node pointer is null
  if (input == nullptr) {
    LOGERROR << "Null pointer input node!";
    return false;
  }

  // Create output
  CombinedTensor* output = new CombinedTensor (input->data.samples(),
      input->data.width(),
      input->data.height(),
      input->data.maps());
  // Tell network about the output
  outputs.push_back (output);

  return true;
}

bool RBFLayer::Connect (const CombinedTensor* input,
                                 CombinedTensor* output) {
  // Check dimensions for equality
  bool valid = input->data.samples() == output->data.samples() &&
               input->data.width() == output->data.width() &&
               input->data.height() == output->data.height() &&
               input->data.maps() == output->data.maps();


	param_ = new CombinedTensor(1);
	param_->data.Clear(1);
	parameters_.push_back(param_);
  return valid;
}

void RBFLayer::FeedForward() {
    // The FeedForward() function computes the layer output given the layer input.
    // This layer only has one single parameter, which we store in r.
	const datum r = param_->data(0);
#pragma omp parallel for default(shared)
    // Our RBF layer performs piecewise "rbf" transformation:
	// rbf(x) = e^-(x*x*r*r)
    // x is the input and r is the parameter of the layer
	for (std::size_t element = 0; element < input_->data.elements(); element++) {
        // Get the value for the input element.
		const datum input_data = input_->data.data_ptr_const()[element];
        // Compute the value of the output element.
		const datum output_data = exp(-(input_data * input_data * r * r));
		output_->data.data_ptr()[element] = output_data;
	}
}

void RBFLayer::BackPropagate() {
    // The BackPropagate() function computes the gradients with respect to
    // the layer input as well as the parameter r and multiplies them 
    // with the respective gradient calculated during back-propagation so far
    // (output_delta).
	const datum r = param_->data(0);
	datum dr = 0;
#pragma omp parallel for default(shared) reduction(+:dr)
	for (std::size_t element = 0; element < input_->data.elements(); element++) {
		const datum input_data = input_->data.data_ptr_const()[element];
		const datum output_delta = output_->delta.data_ptr_const()[element];
		const datum output_data = output_->data.data_ptr_const()[element];

		// rbf'(x) = -2 * r^2 * x * rbf(x)
		// rbf(x) = output
		// This is why we use output_data here (so we don't need to calculate
		// rbf(x) twice).
		// This may be slower for wide networks and large batches
		// because of cache limitations.

		const datum input_delta = (datum)(output_delta * -2.0 * r * r * input_data * output_data);
		const datum r_gradient = (datum)(output_delta * -2.0 * r * input_data * input_data * output_data);

		dr += r_gradient;
		input_->delta.data_ptr()[element] = input_delta;
	}

	param_->delta[0] = dr;
}

void RBFLayer::OnLayerConnect (Layer* next_layer) {
  Conv::Layer::OnLayerConnect (next_layer);
  std::uniform_real_distribution< datum > distribution(-1.0, 1.0);
  param_->data[0] = distribution(generator_);
}


}
