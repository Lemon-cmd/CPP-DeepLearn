#ifndef DIST_DENSE_HPP
#define DIST_DENSE_HPP

namespace dpp {

class DistributedDense : public Layer {

	/* 
	 *
	 * Distributed Dense Layer - A Sequential Normal Forward Layer 
	 *
	 */

	public:
		DistributedDense(const int neurons = 1, const float lrate = 0.001, const std::string activation = "sigmoid", const float erate = 1e-8) {
			assert(neurons > 0 && lrate > 0.0 && erate > 0.0);
			/*
             * Save the parameters, and initalize parallelism and set the appropriate activation function
            */

			Eigen::initParallel();
			
			// set learning and optimization rate; save neurons to output shape
			lrate_ = lrate;
			erate_ = erate;
			output_2d_shape_[1] = neurons;
			
			// bind activation function first to block sigmoid
			act_func_ = std::bind(&DistributedDense::block_sigmoid, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);

			// save the activation name for when saving to a file
			activation_ = activation == "normal" ? activation : activation == "relu" ? activation : activation == "tanh" ? activation : activation == "softmax" ? activation : "sigmoid";  

			// make any necessary changes to activation function
			if (activation == "normal") {
				act_func_ = std::bind(&DistributedDense::block_normal, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
			} else if (activation == "relu") {
				act_func_ = std::bind(&DistributedDense::block_relu, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
			} else if (activation == "tanh") {
				act_func_ = std::bind(&DistributedDense::block_tanh, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
			} else if (activation == "softmax") {
				act_func_ = std::bind(&DistributedDense::block_softmax, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
			}
		}	

		void init(const Eigen::DSizes<ptrdiff_t, 2> input_shape) override {
			assert(input_shape[0] > 0 && input_shape[1] > 0);
			/*
			 *
             * Initialize the parameters of the layer and save the input shape
             *
            */
			
			// set initialized flag
			init_flag_ = true;

			// save input and output shapes
			input_2d_shape_ = input_shape;
			output_2d_shape_[0] = input_shape[0];

			// initialize weights of bias and weight matrices
			bias_ = Eigen::MatrixXf::Ones(1, output_2d_shape_[1]);
			weight_ = Eigen::MatrixXf::Constant(input_shape[1], output_2d_shape_[1], 2.0f / sqrtf(input_shape[1] * output_2d_shape_[1]));
			
			// set the shape of dW, dH, layer delta and layer output 
			doutput_ = Eigen::MatrixXf::Zero(input_shape[0], output_2d_shape_[1]);
			layer_2d_delta_ = Eigen::MatrixXf::Zero(input_shape[0], input_shape[1]);
			layer_2d_output_ = Eigen::MatrixXf::Zero(input_shape[0], output_2d_shape_[1]);

			// select the appropriate size of thread pool and calculate the spread in thread executions
			thread_pool_.resize(input_shape[0] <= std::thread::hardware_concurrency() ? input_shape[0] : input_shape[0] % std::thread::hardware_concurrency() + 1);
			slope_ = input_shape[0] / thread_pool_.size();
		}
		
		void topology() {
			std::cout << "\n\t------- Distributed-Dense Layer --------\n"
					  << "\n\tParameters:\t(" << weight_.rows() * weight_.cols() + bias_.rows() * bias_.cols() << ')'
					  << "\n\tInput-dim:\t(" << input_2d_shape_[0] << ", " << input_2d_shape_[1] << ')'
                      << "\n\tOutput-dim:\t(" << output_2d_shape_[0] << ", " << output_2d_shape_[1] << ")\n"
					  << "\n\t----------------------------------------\n\n";
		}

		void forward(const Eigen::MatrixXf &X) override {
			assert(X.rows() == output_2d_shape_[0] && init_flag_);
			/* *
               * Perform H Calculation sequentially
               * Save X
			   * Loop through sequence
			   * Calculate Z and Activate Z
             * */

			input_ = X;
			for (size_t w = 0; w < output_2d_shape_[0]; w ++)
			{
				layer_2d_output_.row(w) = input_.row(w) * weight_ + bias_;
				act_func_(layer_2d_output_, doutput_, w);
			}
		}

		void update() {
			/* *
			   *
			   * Transpose weight and save it to another matrix
			   * Transpose input and save it to another matrix
			   * Loop through sequence and perform update
			   *
			 * */

			input_T_ = input_.transpose();
			
			old_weight_ = weight_.transpose();

			for (size_t w = 0; w < output_2d_shape_[0]; w ++)
            {
				// calculate dW
                dweight_ = input_T_.col(w) * doutput_.row(w);

				// calculate weight and bias momentums
				vW_ = 0.1 * vW_ + 0.9 * dweight_.array().square().sum();
				vB_ = 0.1 * vB_ + 0.9 * doutput_.row(w).array().square().sum();

				// change bias
				bias_ = bias_.array() - lrate_ / sqrtf(vB_ + erate_) * doutput_.row(w).array();

				// change weight
				weight_ = weight_.array() - lrate_ / sqrtf(vW_ + erate_) * dweight_.array();

				// calculate layer gradient
				layer_2d_delta_.row(w) = doutput_.row(w) * old_weight_;
			}
		}

		void set_delta(const Eigen::MatrixXf &prev_layer_delta) override {
			// dH = dJ (l + 1) * dH
			doutput_ = prev_layer_delta.array() * doutput_.array();
		}

		const std::string name() { return "Distributed-Dense"; }

		const std::string type() { return "2D"; }

		const float MeanSquaredError(const Eigen::MatrixXf &Y, float &accuracy) override {
			// dJ = (H - Y)
			// dH = dJ * dH
			
			doutput_ = (layer_2d_output_.array() - Y.array()) * doutput_.array();
		   	
			accuracy = layer_2d_output_.unaryExpr<float(*)(float)>(&roundf).isApprox(Y) ? accuracy + 1 : accuracy;

			return 0.5 * (layer_2d_output_ - Y).array().square().sum();
		}

		const float CrossEntropyError(const Eigen::MatrixXf &Y, float &accuracy) override {
			/*
             * dJ = (-Y / H)
             * dH = dJ * dH = (H - Y)
             */

			// matches of each row
			float hits = 0.0;
			
			// set new dH
			doutput_ = (layer_2d_output_ - Y);

			// initalize start and end index of thread
			start_ = 0; end_ = slope_;

			// measure accuracy of H concurrently
			for (int t = 0; t < thread_pool_.size(); t ++) {
				thread_pool_[t] = std::thread(&DistributedDense::measure2d, this, std::ref(layer_2d_output_), std::ref(Y), std::ref(hits), start_, end_);
				start_ += slope_; end_ += slope_;
			}

			for (int t = 0; t < thread_pool_.size(); t ++) {
				thread_pool_[t].join();
			}
			
			// perform calculation on the remainder if exists
			if (output_2d_shape_[0] % thread_pool_.size() > 0) {
				measure2d(layer_2d_output_, Y, hits, start_, output_2d_shape_[0]);
			}

			// accuracy = average of matches argmax of H to Y
			accuracy = hits / output_2d_shape_[0] + accuracy;

			return (-Y.array() * layer_2d_output_.array()).log().sum() + 5;
		}
	
		void save(std::ofstream &file) {
			assert(init_flag_);

			// convert bias and weight matrices to vector
			Eigen::Map<Eigen::VectorXf> bias_vect (bias_.data(), bias_.size()), 
									    weight_vect (weight_.data(), weight_.size());

			// write to file
			file << "\nDistributed Dense\n";
			file << "Activation: " << activation_ << "\n";
			file << "Learning_rate: (" << lrate_ << ")\n";
			file << "Momentum_rate: (" << erate_ << ")\n";
			file << "Input_shape: [" << input_2d_shape_[0] << ',' << input_2d_shape_[1] << "]\n";
			file << "Output_shape: [" << output_2d_shape_[0] << ',' << output_2d_shape_[1] << "]\n";	
			file << "Bias: [";
			write_vector(bias_vect, file);
			file << "Weight: [";
			write_vector(weight_vect, file);
		}
	
		void load(const Eigen::DSizes<ptrdiff_t, 2> &input_shape, 
				  const Eigen::DSizes<ptrdiff_t, 2> &output_shape, 
				  const std::vector<float> &bias, 
				  const std::vector<float> &weight) override {
			
			// set the parameters to metadata inside of the layer
			init_flag_ = true;

			// save input and output shapes
			input_2d_shape_ = input_shape; 
			output_2d_shape_ = output_shape;

			// set the shape of dH, layer delta, and layer output
			doutput_ = Eigen::MatrixXf::Zero(input_shape[0], output_2d_shape_[1]);
			layer_2d_delta_ = Eigen::MatrixXf::Zero(input_shape[0], input_shape[1]);
			layer_2d_output_ = Eigen::MatrixXf::Zero(input_shape[0], output_2d_shape_[1]);

			// map bias and weight vectors to their corresponding matrix
			bias_ = Eigen::Map<const Eigen::MatrixXf> (bias.data(), 1, output_2d_shape_[1]);
			weight_ = Eigen::Map<const Eigen::MatrixXf> (weight.data(), input_shape[1], output_2d_shape_[1]);
			
			// set the size of thread pool and calculate the spread in thread executions
			thread_pool_.resize(input_shape[0] <= std::thread::hardware_concurrency() ? input_shape[0] : input_shape[0] % std::thread::hardware_concurrency() + 1);
			slope_ = input_shape[0] / thread_pool_.size();
		}

	private:
		std::string activation_;					// activation name

		float vW_ = 0.0, vB_ = 0.0;					// weight and bias momentums
		
		Eigen::MatrixXf weight_, bias_, doutput_, input_, input_T_, old_weight_, dweight_;			// weight, bias, dH, input, input_transpose, old weight, and dW matrices
		
		std::function <void (Eigen::MatrixXf &, Eigen::MatrixXf &, const size_t &)> act_func_;				// empty activation function
};
}

#endif
