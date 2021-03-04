#ifndef DENSE2D_HPP
#define DENSE2D_HPP

namespace dpp {

class Dense2D : public Layer {

	/*
	 *
	 * Dense Layer 2D
	 *
	 */

	public:
		Dense2D(const int neurons = 1, const float lrate = 0.001, const std::string activation = "normal", const float erate = 1e-8) {
			assert (neurons > 0 && lrate > 0.0 && erate > 0.0);
			/*
             * Save the parameters, and initalize parallelism and set the appropriate activation and recurrent activation functions
             */

			Eigen::initParallel();
			
			// set learning and optimization rate
			lrate_ = lrate;
			erate_ = erate;

			// save neurons to output shape
			output_3d_shape_[0] = 1;
			output_3d_shape_[2] = neurons;

			// bind activation function to no activation first
			act_func_ = std::bind(&Dense2D::tensor_normal, this, std::placeholders::_1, std::placeholders::_2);

			// save activation name
			activation_ = activation == "normal" ? activation : activation == "tanh" ? activation : activation == "softmax" ? activation : activation == "relu" ? activation : "sigmoid"; 

			// bind activation function to another function if appropriate
			if (activation == "relu") {
				act_func_ = std::bind(&Dense2D::tensor_relu, this, std::placeholders::_1, std::placeholders::_2);
			} else if (activation == "tanh") {
				act_func_ = std::bind(&Dense2D::tensor_tanh, this, std::placeholders::_1, std::placeholders::_2);
			} else if (activation == "sigmoid") {
				act_func_ = std::bind(&Dense2D::tensor_sigmoid, this, std::placeholders::_1, std::placeholders::_2);
			} else if (activation == "softmax") {
				act_func_ = std::bind(&Dense2D::tensor_softmax, this, std::placeholders::_1, std::placeholders::_2);
			}
		}

		void init(const Eigen::DSizes<ptrdiff_t, 3> input_shape) override {
			assert(input_shape[0] == 1 && input_shape[1] > 0 && input_shape[2] > 0);
			/*
             *
             * Initialize the parameters of the layer and save the input shape
             *
             */

			// set initialized flag
			init_flag_ = true;

			// save input and calculate output shape
			input_3d_shape_ = input_shape;
			output_3d_shape_[1] = input_shape[1];

			// set shape of bias and weight tensors
			bias_ = Eigen::Tensor<float, 3> {1, input_shape[1], output_3d_shape_[2]};
			weight_ = Eigen::Tensor<float, 3> {1, input_shape[2], output_3d_shape_[2]};

			// set shape of relu comparison matrices
			relu_3d_ones_ = Eigen::Tensor<float, 3> {1, input_shape[1], output_3d_shape_[2]};
			relu_3d_zeroes_ = Eigen::Tensor<float, 3> {1, input_shape[1], output_3d_shape_[2]};

			// set the values of above matrices
			bias_.setConstant(1.0);
			relu_3d_ones_.setConstant(1.0);
			relu_3d_zeroes_.setConstant(0.0);
			weight_.setConstant(2.0 / sqrtf(input_shape[2] * output_3d_shape_[2]));
		}

		void topology() {
			std::cout << "\n\t------------ 2D-Dense Layer ------------\n"
					  << "\n\tParameters:\t(" << weight_.dimension(1) * weight_.dimension(2) + bias_.dimension(1) * bias_.dimension(2) << ')'
					  << "\n\tInput-dim:\t(" << input_3d_shape_[0] << ", " << input_3d_shape_[1] << ", " << input_3d_shape_[2] << ')'
                      << "\n\tOutput-dim:\t(" << output_3d_shape_[0] << ", " << output_3d_shape_[1] << ", " << output_3d_shape_[2] << ")\n"
					  << "\n\t----------------------------------------\n\n";
		}

		void forward(const Eigen::Tensor<float, 3> &X) override {
			assert(X.dimension(0) == input_3d_shape_[0] && X.dimension(1) == input_3d_shape_[1] && X.dimension(2) == input_3d_shape_[2] && init_flag_);
			/*
			 * Check X-dimension
			 * Apply forward function
			 * Activate H
			*/

			input_ = X;
			layer_3d_output_ = input_.contract(weight_, output_dot_dim_).reshape(output_3d_shape_) + bias_;
			act_func_(layer_3d_output_, doutput_);
		}

		void update() {
			// calculate bias momentum
			vB_ = 0.1 * vB_ + 0.9 * ((Eigen::Tensor<float, 0>) doutput_.square().sum())(0);

			// change bias
			bias_ = bias_ - lrate_ / sqrtf(vB_ + erate_) * doutput_;

			// calculate dW
			dweight_ = input_.contract(doutput_, dweight_dot_dim_).reshape(weight_.dimensions());

			// calculate weight momentum
			vW_ = 0.1 * vW_ + 0.9 * ((Eigen::Tensor<float, 0>) dweight_.square().sum())(0);

			// change weight
			weight_ = weight_ - lrate_ / sqrtf(vW_ + erate_) * dweight_;

			// calculate layer delta
			layer_3d_delta_ = doutput_.contract(weight_, delta_dot_dim_).reshape(input_.dimensions());
		}

		void set_delta(const Eigen::Tensor<float, 3> &prev_layer_delta) override {
			// dH = dJ (l + 1) * dH
			doutput_ = prev_layer_delta * doutput_;
		}

		const std::string name() { return "Dense2D"; }

		const std::string type() { return "3D"; }

		const float MeanSquaredError(const Eigen::Tensor<float, 3> &Y, float &accuracy) override {
			// dH = (H - Y) * dH
			doutput_ = (layer_3d_output_ - Y) * doutput_;

			Eigen::Map<const Eigen::VectorXf> YH(Y.data(), Y.size());
			Eigen::Map<const Eigen::VectorXf> H(layer_3d_output_.data(), layer_3d_output_.size());

			accuracy = H.isApprox(YH, 0.1) ? accuracy + 1 : accuracy;
            return ((Eigen::Tensor<float, 0>) (0.5 * (layer_3d_output_ - Y).square().sum()))(0);
		}

		const float CrossEntropyError(const Eigen::Tensor<float, 3> &Y, float &accuracy) override {
			float hits = 0.0;
			
			doutput_ = layer_3d_output_ - Y;

			measure3d(layer_3d_output_, Y, hits, 0, output_3d_shape_[1]);
			
			accuracy = hits / output_3d_shape_[1] + accuracy;
			
			return ((Eigen::Tensor<float, 0>) ((-Y * layer_3d_output_.log()).sum()))(0);
        }

		void save(std::ofstream &file) {
			assert(init_flag_);

			// convert bias and weight into vector
			Eigen::Map<Eigen::VectorXf> bias_vect (bias_.data(), bias_.size()), 
										weight_vect (weight_.data(), weight_.size());

			// write to file
			file << "\nDense2D\n";
			file << "Activation: " << activation_ << '\n';
			file << "Learning_rate: (" << lrate_ << ")\n";
			file << "Momentum_rate: (" << erate_ << ")\n";
			file << "Input_shape: [" << input_3d_shape_[0] << ',' << input_3d_shape_[1] << ',' << input_3d_shape_[2] << "]\n";
			file << "Output_shape: [" << output_3d_shape_[0] << ',' << output_3d_shape_[1] << ',' << output_3d_shape_[2] << "]\n";
			file << "Bias: [";
			write_vector(bias_vect, file);
			file << "Weight: [";
			write_vector(weight_vect, file);
		}

		void load(const Eigen::DSizes<ptrdiff_t, 3> &input_shape, 
				  const Eigen::DSizes<ptrdiff_t, 3> &output_shape, 
				  std::vector<float> &bias, 
				  std::vector<float> &weight) override {
		
			// set the parameters to metadata inside of the layer	
			init_flag_ = true;

			// save input and output shape
            input_3d_shape_ = input_shape;
            output_3d_shape_[1] = input_shape[1];			
			
			// set relu comparison matrices' shape
			relu_3d_ones_ = Eigen::Tensor<float, 3> {1, input_shape[1], output_3d_shape_[2]};
            relu_3d_zeroes_ = Eigen::Tensor<float, 3> {1, input_shape[1], output_3d_shape_[2]};

			// set their values
            relu_3d_ones_.setConstant(1.0);
            relu_3d_zeroes_.setConstant(0.0);

			// map bias and weight vectors to their corresponding matrix
			bias_ = Eigen::TensorMap <Eigen::Tensor<float, 3>> (bias.data(), output_3d_shape_);
			weight_ = Eigen::TensorMap <Eigen::Tensor<float, 3>> (weight.data(), Eigen::DSizes<ptrdiff_t, 3> {1, input_shape[2], output_3d_shape_[2]});
		}

	private:

		std::string activation_;		// activation name
		
		float vW_ = 0.0, vB_ = 0.0;		// momentums of weight and bias
		
		Eigen::Tensor<float, 3> input_, doutput_, weight_, dweight_, bias_;		// input, dH, weight, dW, and bias matrices
		
		// index pair for tensor contractions
		const Eigen::array<Eigen::IndexPair<int>, 1> output_dot_dim_ {Eigen::IndexPair<int> {2, 1}}, 
													 delta_dot_dim_ {Eigen::IndexPair<int> {2, 2}},
													 dweight_dot_dim_ {Eigen::IndexPair<int> {1, 1}}; 

		std::function <void (Eigen::Tensor<float, 3> &, Eigen::Tensor<float, 3> &)> act_func_; // empty function for activation
};

}

#endif
