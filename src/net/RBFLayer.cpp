/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
#include <vector>

#include "CombinedTensor.h"
#include "Log.h"

#include "RBFLayer.h"

namespace Conv {
  
RBFLayer::RBFLayer() {
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

  return valid;
}

void RBFLayer::FeedForward() {
	const datum r = 0.5;
#pragma omp parallel for default(shared)
	for (std::size_t element = 0; element < input_->data.elements(); element++) {
		const datum input_data = input_->data.data_ptr_const()[element];

		// Calculate rbf: rbf(x) = e^-(x*x*r*r)
		const datum output_data = exp(-(input_data * input_data * r * r));
		output_->data.data_ptr()[element] = output_data;
	}
}

void RBFLayer::BackPropagate() {
	const datum r = 0.5;
#pragma omp parallel for default(shared)
	for (std::size_t element = 0; element < input_->data.elements(); element++) {
		const datum input_data = input_->data.data_ptr_const()[element];
		const datum output_delta = output_->delta.data_ptr_const()[element];
		const datum output_data = output_->data.data_ptr_const()[element];

		// rbf'(x) = -2 * r^2 * x * rbf(x)
		// rbf(x) = output
		// this is why we use output_data here (so we don't need to calculate
		// rbf(x) twice).
		// This may be slower for wide networks and large batches
		// because of cache limitations.

		const datum input_delta = (datum)(output_delta * -2.0 * r * r * input_data * output_data);
		input_->delta.data_ptr()[element] = input_delta;
	}
}

}
