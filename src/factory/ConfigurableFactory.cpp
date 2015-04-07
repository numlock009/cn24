/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */
#include <cstdio>

#include "ErrorLayer.h"

#include "ConvolutionLayer.h"
#include "ResizeLayer.h"
#include "MaxPoolingLayer.h"
#include "NonLinearityLayer.h"
#include "UpscaleLayer.h"
#include "SpatialPriorLayer.h"
#include "ConfigParsing.h"
#include "NetGraph.h"

#include "ConfigurableFactory.h"

namespace Conv {



ConfigurableFactory::ConfigurableFactory (std::istream& file, const unsigned int seed, bool is_training_factory) :
  seed_ (seed), file_ (file), method_ (FCN) {
  file_.clear();
  file_.seekg (0, std::ios::beg);

  receptive_field_x_ = 0;
  receptive_field_y_ = 0;

  patch_field_x_ = 0;
  patch_field_y_ = 0;
  
  bool ignore_layers = false;

  // Calculate patch size / receptive field size
  while (! file_.eof()) {
    std::string line;
    std::getline (file_, line);

    std::string method;
    ParseStringParamIfPossible (line, "method", method);

    if (method.compare (0, 5, "patch") == 0) {
      if (is_training_factory) {
        method_ = PATCH;
        LOGINFO << "Setting method to PATCH";
      }
    }
    
    if (line.compare (0, 6, "manual") == 0) {
      unsigned int rfx = 0, rfy = 0, fx = 0, fy = 0;
      ParseCountIfPossible(line, "rfx", rfx);
      ParseCountIfPossible(line, "rfy", rfy);
      ParseCountIfPossible(line, "factorx", fx);
      ParseCountIfPossible(line, "factory", fy);
      if(rfx > 0 && rfy > 0) {
        LOGINFO << "Using manual receptive field method";
        receptive_field_x_ = rfx;
        receptive_field_y_ = rfy;
        factorx = fx;
        factory = fy;
        ignore_layers = true;
      }
    }

    if (line.compare (0, 1, "?") == 0 && !ignore_layers) {
      line = line.substr (1);

      if (StartsWithIdentifier (line, "convolutional")) {
        unsigned int kx, ky;
        ParseKernelSizeIfPossible (line, "size", kx, ky);
        LOGDEBUG << "Adding convolutional layer to receptive field (" << kx << "," << ky << ")";
        receptive_field_x_ += factorx * (kx - 1);
        receptive_field_y_ += factory * (ky - 1);
      }

      if (StartsWithIdentifier (line, "maxpooling")) {
        unsigned int kx, ky;
        LOGDEBUG << "Convolutional layer";
        ParseKernelSizeIfPossible (line, "size", kx, ky);
        LOGDEBUG << "Adding maxpooling layer to receptive field (" << kx << "," << ky << ")";
        factorx *= kx;
        factory *= ky;
      }
    }
  }

  if (method_ == PATCH) {
    receptive_field_x_ += factorx;
    receptive_field_y_ += factory;
  }

  patch_field_x_ = receptive_field_x_ + factorx;
  patch_field_y_ = receptive_field_y_ + factory;
}

Layer* ConfigurableFactory::CreateLossLayer (const unsigned int output_classes) {
  return new ErrorLayer();
}

int ConfigurableFactory::AddLayers (Net& net, Connection data_layer_connection, const unsigned int output_classes, bool add_loss_layer, std::ostream& graph_output) {
  std::mt19937 rand (seed_);
  file_.clear();
  file_.seekg (0, std::ios::beg);
  int last_layer_output = data_layer_connection.output;
  int last_layer_id = data_layer_connection.net;

	graph_output << "node" << last_layer_id << " [shape=record, label=\"" << 
		"{Dataset Input | {<o0> Data | <o1> Label | <o2> Helper | <o3> Weight}}"
		<< "\"];\n";

  int current_receptive_field_x = patch_field_x_;
  int current_receptive_field_y = patch_field_y_;
  datum llr_factor = 1.0;
  
	
  Connection stack_a[64];
  Connection stack_b[64];
  int stack_a_pos = -1;
  int stack_b_pos = -1;

  if (method_ == FCN) {
    /*Tensor* const net_output = &net.buffer (data_layer_connection.net, data_layer_connection.output)->data;
    datum net_output_size_x = net_output->width();
    datum net_output_size_y = net_output->height();
    llr_factor /= net_output_size_x;
    llr_factor /= net_output_size_y;
#ifdef CN24_EMULATE_PATCH_LEARNING
    llr_factor /= patch_field_x_;
    llr_factor /= patch_field_y_;
#endif
    LOGINFO << "Local learning rate factor is (initially): " << llr_factor;*/

		int input_layer_id = last_layer_id;
		int input_layer_output = last_layer_output;
		ResizeLayer* rl = new ResizeLayer(receptive_field_x_, receptive_field_y_);
    last_layer_id = net.AddLayer (rl, { data_layer_connection });
    last_layer_output = 0;

		WriteNode(graph_output, rl, input_layer_id, input_layer_output, last_layer_id, 1);
  }

  bool first_layer = true;

  while (! file_.eof()) {
    std::string line;
    std::getline (file_, line);

    /*
     * PREPROCESSING
     */
    
    // Replace number of output neurons
    if (line.find ("(o)") != std::string::npos) {
      char buf[64];
      sprintf (buf, "%d", output_classes);
      line.replace (line.find ("(o)"), 3, buf);
    }

    // Replace fully connected layers
    if (line.find ("fullyconnected") != std::string::npos) {
      line.replace (line.find ("fullyconnected"), 14, "convolutional size=1x1");
      line.replace (line.find ("neurons="), 8, "kernels=");
    }

    if (method_ == FCN) {
      // Remove flatten layers
      if (line.find ("flatten") != std::string::npos) {
        line = "";
      }
    }
    
    if (line.compare ("?output") == 0) {
      if (output_classes == 1) {
        line = "?tanh";
      } else {
        line = "?sigm";
      }
    }
    
    /*
     * STACK OPERATIONS
     */
    if (line.compare("pusha") == 0) {
      stack_a[++stack_a_pos].net = last_layer_id;
      stack_a[stack_a_pos].output = last_layer_output;
    }

    if (line.compare("pushb") == 0) {
      stack_b[++stack_b_pos].net = last_layer_id;
      stack_b[stack_b_pos].output = last_layer_output;
    }
    
    if (line.compare("popa") == 0) {
      last_layer_id = stack_a[stack_a_pos].net;
      last_layer_output = stack_a[stack_a_pos--].output;
    }
    
    if (line.compare("popb") == 0) {
      last_layer_id = stack_b[stack_b_pos].net;
      last_layer_output = stack_b[stack_b_pos--].output;
    }
    
    /*
     * PARSING
     */
    if (line.compare (0, 1, "?") == 0) {
      line = line.substr (1);
      LOGDEBUG << "Parsing layer: " << line;

      if (StartsWithIdentifier (line, "convolutional")) {
        unsigned int kx = 1, ky = 1, k = 1;
        datum llr = 1;
        datum dropout_fraction = 0.0;
        ParseKernelSizeIfPossible (line, "size", kx, ky);
        ParseCountIfPossible (line, "kernels", k);
        ParseDatumParamIfPossible (line, "dropout", dropout_fraction);
        ParseDatumParamIfPossible (line, "llr", llr);
        LOGDEBUG << "Parsed dropout fraction: " << dropout_fraction;

        ConvolutionLayer* cl = new ConvolutionLayer (kx, ky, k, rand(), dropout_fraction);

        // if (method_ == FCN) {
          LOGDEBUG << "LLR factor: " << llr_factor << ", RFX: " << current_receptive_field_x;
        /*  cl->SetLocalLearningRate (llr * llr_factor * (datum) current_receptive_field_x * (datum) current_receptive_field_y);
#ifdef CN24_EMULATE_PATCH_LEARNING
          current_receptive_field_x -= (kx - 1);
          current_receptive_field_y -= (ky - 1);
#endif
        } else { */
          cl->SetLocalLearningRate (llr * llr_factor);
        /* } */

        if (first_layer)
          cl->SetBackpropagationEnabled (false);
				int input_layer_id = last_layer_id;
				int input_layer_output = last_layer_output;
        last_layer_id = net.AddLayer (cl ,
        { Connection (last_layer_id, last_layer_output) });
        last_layer_output = 0;
        first_layer = false;

				WriteNode(graph_output, cl, input_layer_id, input_layer_output, last_layer_id, 1);
      }

      if (StartsWithIdentifier (line, "maxpooling")) {
        unsigned int kx = 1, ky = 1;
        ParseKernelSizeIfPossible (line, "size", kx, ky);

        /*if (method_ == FCN) {
#ifdef CN24_EMULATE_PATCH_LEARNING
          current_receptive_field_x /= kx;
          current_receptive_field_y /= ky;
          llr_factor *= (datum) (kx * ky);
#endif
        }*/

				int input_layer_id = last_layer_id;
				int input_layer_output = last_layer_output;

        MaxPoolingLayer* mp = new MaxPoolingLayer (kx, ky);
        last_layer_id = net.AddLayer (mp ,
        { Connection (last_layer_id, last_layer_output) });
        last_layer_output = 0;

				WriteNode(graph_output, mp, input_layer_id, input_layer_output, last_layer_id, 1);
      }

      if (StartsWithIdentifier (line, "sigm")) {
				int input_layer_id = last_layer_id;
				int input_layer_output = last_layer_output;

        SigmoidLayer* l = new SigmoidLayer();
        last_layer_id = net.AddLayer (l ,
        { Connection (last_layer_id, last_layer_output) });
        last_layer_output = 0;

				WriteNode(graph_output, l, input_layer_id, input_layer_output, last_layer_id, 1);
      }

      if (StartsWithIdentifier (line, "relu")) {
				int input_layer_id = last_layer_id;
				int input_layer_output = last_layer_output;

        ReLULayer* l = new ReLULayer();
        last_layer_id = net.AddLayer (l ,
        { Connection (last_layer_id, last_layer_output) });
        last_layer_output = 0;

				WriteNode(graph_output, l, input_layer_id, input_layer_output, last_layer_id, 1);
      }

      if (StartsWithIdentifier (line, "tanh")) {
				int input_layer_id = last_layer_id;
				int input_layer_output = last_layer_output;

        TanhLayer* l = new TanhLayer();
        last_layer_id = net.AddLayer (l ,
        { Connection (last_layer_id, last_layer_output) });
        last_layer_output = 0;

				WriteNode(graph_output, l, input_layer_id, input_layer_output, last_layer_id, 1);
      }

      if (StartsWithIdentifier (line, "spatialprior")) {
        if (method_ == FCN) {
					int input_layer_id = last_layer_id;
					int input_layer_output = last_layer_output;

          SpatialPriorLayer* l = new SpatialPriorLayer();
          last_layer_id = net.AddLayer (l ,
          { Connection (last_layer_id, last_layer_output) });
          last_layer_output = 0;

				WriteNode(graph_output, l, input_layer_id, input_layer_output, last_layer_id, 1);
        }
      }
    }
  }

  if (method_ == FCN && (factorx != 1 || factory != 1)) {
		int input_layer_id = last_layer_id;
		int input_layer_output = last_layer_output;

    last_layer_id = net.AddLayer (new UpscaleLayer (factorx, factory),
    { Connection (last_layer_id, last_layer_output) });
    last_layer_output = 0;
    LOGDEBUG << "Added upscaling layer for FCN";

		graph_output << "node" << last_layer_id << " [shape=record, shape=record, label=\"" <<
			"{Upscale Layer (" << factorx << "x" << factory  << ") | <o0> Output" << "}\"];\n";
		graph_output << "node" << input_layer_id << ":o" << input_layer_output
			<< " -> node" << last_layer_id << ";\n";
  }

	// Add loss layer
	if (add_loss_layer) {
		int loss_layer_id = net.AddLayer(CreateLossLayer(output_classes), {
			Connection(last_layer_id, last_layer_output),
			Connection(data_layer_connection.net, 1),
			Connection(data_layer_connection.net, 3)
		});


		graph_output << "node" << loss_layer_id << " [shape=record, label=\"" <<
			"{Loss Layer | <o0> Output}" << "\"];\n";

		graph_output << "node" <<  last_layer_id << ":o" << last_layer_output
			<< " -> node" << loss_layer_id << ";\n";
		graph_output << "node" <<  data_layer_connection.net << ":o" << 1
			<< " -> node" << loss_layer_id << ";\n";
		graph_output << "node" <<  data_layer_connection.net << ":o" << 3
			<< " -> node" << loss_layer_id << ";\n";
	}

  return last_layer_id;
}

bool ConfigurableFactory::AddLayers(NetGraph& net, NetGraphConnection data_layer_connection, const unsigned int output_classes, bool add_loss_layer) {
  std::mt19937 rand (seed_);
  file_.clear();
  file_.seekg (0, std::ios::beg);

	NetGraphConnection last_connection = data_layer_connection;

  int current_receptive_field_x = patch_field_x_;
  int current_receptive_field_y = patch_field_y_;
	
  NetGraphConnection stack_a[64];
  NetGraphConnection stack_b[64];
  int stack_a_pos = -1;
  int stack_b_pos = -1;

  if (method_ == FCN) {
		ResizeLayer* rl = new ResizeLayer(receptive_field_x_, receptive_field_y_);
		NetGraphNode* node = new NetGraphNode(rl, last_connection);
		net.AddNode(node);

		last_connection.node = node;
		last_connection.buffer = 0;
  }

  bool first_layer = true;

  while (! file_.eof()) {
    std::string line;
    std::getline (file_, line);

    /*
     * PREPROCESSING
     */
    
    // Replace number of output neurons
    if (line.find ("(o)") != std::string::npos) {
      char buf[64];
      sprintf (buf, "%d", output_classes);
      line.replace (line.find ("(o)"), 3, buf);
    }

    // Replace fully connected layers
    if (line.find ("fullyconnected") != std::string::npos) {
      line.replace (line.find ("fullyconnected"), 14, "convolutional size=1x1");
      line.replace (line.find ("neurons="), 8, "kernels=");
    }

    if (method_ == FCN) {
      // Remove flatten layers
      if (line.find ("flatten") != std::string::npos) {
        line = "";
      }
    }
    
    if (line.compare ("?output") == 0) {
      if (output_classes == 1) {
        line = "?tanh";
      } else {
        line = "?sigm";
      }
    }
    
    /*
     * STACK OPERATIONS
     */
    if (line.compare("pusha") == 0) {
      stack_a[++stack_a_pos] = last_connection;
    }

    if (line.compare("pushb") == 0) {
      stack_b[++stack_b_pos] = last_connection;
    }
    
    if (line.compare("popa") == 0) {
      last_connection = stack_a[stack_a_pos];
    }
    
    if (line.compare("popb") == 0) {
      last_connection = stack_b[stack_b_pos];
    }
    
    /*
     * PARSING
     */
    if (line.compare (0, 1, "?") == 0) {
      line = line.substr (1);
      LOGDEBUG << "Parsing layer: " << line;

      if (StartsWithIdentifier (line, "convolutional")) {
        unsigned int kx = 1, ky = 1, k = 1;
        datum llr = 1;
        datum dropout_fraction = 0.0;
        ParseKernelSizeIfPossible (line, "size", kx, ky);
        ParseCountIfPossible (line, "kernels", k);
        ParseDatumParamIfPossible (line, "dropout", dropout_fraction);
        ParseDatumParamIfPossible (line, "llr", llr);
        LOGDEBUG << "Parsed dropout fraction: " << dropout_fraction;

        ConvolutionLayer* cl = new ConvolutionLayer (kx, ky, k, rand(), dropout_fraction);
				cl->SetLocalLearningRate (llr);

				NetGraphNode* node = new NetGraphNode(cl, last_connection);
				net.AddNode(node);
				last_connection.buffer = 0;
				last_connection.node = node;
      }

      if (StartsWithIdentifier (line, "maxpooling")) {
        unsigned int kx = 1, ky = 1;
        ParseKernelSizeIfPossible (line, "size", kx, ky);

        MaxPoolingLayer* mp = new MaxPoolingLayer (kx, ky);

				NetGraphNode* node = new NetGraphNode(mp, last_connection);
				net.AddNode(node);
				last_connection.buffer = 0;
				last_connection.node = node;
      }

      if (StartsWithIdentifier (line, "sigm")) {
        SigmoidLayer* l = new SigmoidLayer();
				NetGraphNode* node = new NetGraphNode(l, last_connection);
				net.AddNode(node);
				last_connection.buffer = 0;
				last_connection.node = node;
      }

      if (StartsWithIdentifier (line, "relu")) {
        ReLULayer* l = new ReLULayer();
				NetGraphNode* node = new NetGraphNode(l, last_connection);
				net.AddNode(node);
				last_connection.buffer = 0;
				last_connection.node = node;
      }

      if (StartsWithIdentifier (line, "tanh")) {
        TanhLayer* l = new TanhLayer();
				NetGraphNode* node = new NetGraphNode(l, last_connection);
				net.AddNode(node);
				last_connection.buffer = 0;
				last_connection.node = node;
      }

      if (StartsWithIdentifier (line, "spatialprior")) {
        if (method_ == FCN) {
          SpatialPriorLayer* l = new SpatialPriorLayer();
					NetGraphNode* node = new NetGraphNode(l, last_connection);
					net.AddNode(node);
					last_connection.buffer = 0;
					last_connection.node = node;
        }
      }
    }
  }

  if (method_ == FCN && (factorx != 1 || factory != 1)) {
		UpscaleLayer* l = new UpscaleLayer(factorx, factory);
		NetGraphNode* node = new NetGraphNode(l, last_connection);
		net.AddNode(node);
		last_connection.buffer = 0;
		last_connection.node = node;
		
    LOGDEBUG << "Added upscaling layer for FCN";
  }

	// Add loss layer
	if (add_loss_layer) {
		// Collect inputs
		NetGraphConnection label_connection = data_layer_connection;
		label_connection.buffer = 1;

		NetGraphConnection weight_connection = data_layer_connection;
		weight_connection.buffer = 3;

		Layer* loss_layer = CreateLossLayer(output_classes);
		NetGraphNode* node = new NetGraphNode(loss_layer);
		node->input_connections.push_back(last_connection);
		node->input_connections.push_back(label_connection);
		node->input_connections.push_back(weight_connection);

		net.AddNode(node);
	}

	return net.IsComplete();
}

void ConfigurableFactory::InitOptimalSettings() {
  file_.clear();
  file_.seekg (0, std::ios::beg);

  while (!file_.eof()) {
    std::string line;
    std::getline (file_, line);

    ParseDatumIfPossible (line, "l1", optimal_settings_.l1_weight);
    ParseDatumIfPossible (line, "l2", optimal_settings_.l2_weight);
    ParseDatumIfPossible (line, "lr", optimal_settings_.learning_rate);
    ParseDatumIfPossible (line, "gamma", optimal_settings_.gamma);
    ParseDatumIfPossible (line, "momentum", optimal_settings_.momentum);
    ParseDatumIfPossible (line, "exponent", optimal_settings_.exponent);
    ParseUIntIfPossible (line, "iterations", optimal_settings_.iterations);
    ParseUIntIfPossible (line, "sbatchsize", optimal_settings_.sbatchsize);
    ParseUIntIfPossible (line, "pbatchsize", optimal_settings_.pbatchsize);
  }
}

void ConfigurableFactory::WriteNode(std::ostream& graph_output, Layer* layer, int source_id, int source_port, int node_id, int outputs) {
	graph_output << "node" << node_id << " [shape=record, label=\"" <<
		"{" << layer->GetLayerDescription();
	if (outputs > 1) {
		graph_output << " | {";
		for (int i = 0; i < outputs; i++) {
			if (i == 0)
				graph_output << " | ";
			graph_output << "<o" << i << "> Output " << i;
		}
		graph_output << "}";
	}
	else if (outputs == 1) {
		graph_output << " | <o0> Output";
	} 
	graph_output << " }\"];\n";
	graph_output << "node" << source_id << ":o" << source_port
		<< " -> node" << node_id << ";\n";
}

}
