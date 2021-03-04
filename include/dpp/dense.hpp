#ifndef DENSE_HPP
#define DENSE_HPP

namespace dpp {

class Dense : public Layer {

	/* 
	 * 
	 * Dense Layer or Normal Forward Layer
	 *
	 */

	public:
		Dense(const int neurons = 1, const float lrate = 0.001, const std::string activation = "sigmoid", const float erate = 1e-8) {
			assert(neurons > 0 && lrate > 0.0 & erate > 0.0);
			/* 
			 * Save the parameters, and initalize parallelism and set the appropriate activation function 
			*/
			
			Eigen::initParallel();		
			
			// set learning and optimization rate; save neurons to output shape
			lrate_ = lrate;
			erate_ = erate;
			output_2d_shape_[1] = neurons;	
			
			// bind activation function first to sigmoid
			act_func_ = std::bind(&Dense::matrix_sigmoid, this, std::placeholders::_1, std::placeholders::_2);

			// save the activation name for when saving to a file 
			activation_ = activation == "normal" ? activation : activation == "tanh" ? activation : activation == "softmax" ? activation : activation == "relu" ? activation : "sigmoid"; 

			// make any necessary changes to activation function
			if (activation == "normal") {
				act_func_ = std::bind(&Dense::matrix_normal, this, std::placeholders::_1, std::placeholders::_2);
			} else if (activation == "relu") {
				act_func_ = std::bind(&Dense::matrix_relu, this, std::placeholders::_1, std::placeholders::_2);
			} else if (activation == "tanh") {
				act_func_ = std::bind(&Dense::matrix_tanh, this, std::placeholders::_1, std::placeholders::_2);
			} else if (activation == "softmax"){
				act_func_ = std::bind(&Dense::matrix_softmax, this, std::placeholders::_1, std::placeholders::_2);
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
			bias_ = Eigen::MatrixXf::Ones(input_shape[0], output_2d_shape_[1]);
			weight_ = Eigen::MatrixXf::Constant(input_shape[1], output_2d_shape_[1], 2.0f / sqrtf(input_shape[1] * output_2d_shape_[1]));
			
			// select the appropriate size of thread pool and calculate the spread in thread executions
			thread_pool_.resize(input_shape[0] <= std::thread::hardware_concurrency() ? input_shape[0] : input_shape[0] % std::thread::hardware_concurrency() + 1);
			slope_ = input_shape[0] / thread_pool_.size();
		}

		void topology() {
			std::cout << "\n\t-------------- Dense Layer -------------\n"
					  << "\n\tParameters:\t(" << weight_.rows() * weight_.cols() + bias_.rows() * bias_.cols() << ')'
					  << "\n\tInput-dim:\t(" << input_2d_shape_[0] << ", " << input_2d_shape_[1] << ')'
                      << "\n\tOutput-dim:\t(" << output_2d_shape_[0] << ", " << output_2d_shape_[1] << ")\n"
					  << "\n\t----------------------------------------\n\n";
		}

		void forward(const Eigen::MatrixXf &X) override {
			assert(X.rows() == input_2d_shape_[0] && X.cols() == input_2d_shape_[1] && init_flag_);
			/* *
			 * Perform H Calculation
			 * Save X
			 * Calculate Z
			 * Activate Z
			 * */

			input_ = X;
			layer_2d_output_ = input_ * weight_ + bias_;
			act_func_(layer_2d_output_, doutput_);
		}

		void update() {
			/* *
			 * Perform Update 
			 * dW = X.T * dH * dL
			 * dB = dH * dL
			 * Change Momentum of dW and dB
			 * Change W and B
			 * */
		
			// calculate gradient of weight	
			dweight_ = input_.transpose() * doutput_; 

			// calculate the momentums of bias and weights based on Adam prop.
			vB_ = 0.1 * vB_ + 0.9 * doutput_.array().square().sum();
			vW_ = 0.1 * vW_ + 0.9 * dweight_.array().square().sum();
			
			// change bias
			bias_ = bias_.array() - lrate_ / sqrtf(vB_ + erate_) * doutput_.array();

			// calculate the gradient of the layer before changing its weight
			layer_2d_delta_ = doutput_ * weight_.transpose();
			
			// change weight	
			weight_ = weight_.array() - lrate_ / sqrtf(vW_ + erate_) * dweight_.array();
		}

		void set_delta(const Eigen::MatrixXf& prev_layer_delta) override {
			// multiply dL (l + 1) to dH (l)
			doutput_ = prev_layer_delta.array() * doutput_.array();
		}
		
		const std::string name() { return "Dense"; }

		const std::string type() { return "2D"; }

		const float MeanSquaredError(const Eigen::MatrixXf &Y, float &accuracy) override {
			/* 
			 * dL = (H - Y) based on derivative of MSquaredError
			 * dH = dL * dH 
			 * Calculate Accuracy and Loss
			*/
			
			layer_2d_delta_ = (layer_2d_output_ - Y);

			doutput_ = layer_2d_delta_.array() * doutput_.array();

		   	accuracy = layer_2d_output_.unaryExpr<float(*)(float)>(&roundf).isApprox(Y) ? accuracy + 1 : accuracy;

			return 0.5 * layer_2d_delta_.array().square().sum();
		}
		
		const float CrossEntropyError(const Eigen::MatrixXf &Y, float &accuracy) override {
			/* 
			 * dH = -Y / H 
			 * dL * dH = (-Y / H) * (H * (1 - H)) = H - Y 
			 * Calculate Accuracy and Loss 
			*/
			
			// matches of each row	
			float hits = 0.0;
			
			// set new dH	
			doutput_ = (layer_2d_output_ - Y);

			// initalize start and end index of thread 
			start_ = 0; end_ = slope_;
			
			// measure accuracy of H concurrently
			for (int t = 0; t < thread_pool_.size(); t ++) {
				thread_pool_[t] = std::thread(&Dense::measure2d, this, std::ref(layer_2d_output_), std::ref(Y), std::ref(hits), start_, end_);
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
			
			return (-Y.array() * layer_2d_output_.array()).log().sum();
		}
		
		void save(std::ofstream &file) {
			assert(init_flag_);

			// convert the bias and weight into vectors
			Eigen::Map<Eigen::VectorXf> bias_vect (bias_.data(), bias_.size()), 
										weight_vect (weight_.data(), weight_.size());
			
			// write to file 
			file << "\nDense\n";
			file << "Activation: " << activation_ << '\n';
			file << "Learning_rate: (" << lrate_ << ")\n";
			file << "Momentum_rate: (" << erate_ << ")\n";
			file << "Input_shape: [" << input_2d_shape_[0] << ',' << input_2d_shape_[1] << "]\n";
			file << "Output_shape: [" << output_2d_shape_[0] << ',' << output_2d_shape_[1] << "]\n";
			file << "Bias: [";
			write_vector(bias_vect, file);	
			file << "Weight: [";	
			write_vector(weight_vect, file);
		}
	
		void load (const Eigen::DSizes<ptrdiff_t, 2> &input_shape, 
				   const Eigen::DSizes<ptrdiff_t, 2> &output_shape,
                   const std::vector<float> &bias, 
				   const std::vector<float> &weight) override {
			
			// set the parameters to metadata inside of the layer	
			init_flag_ = true;

			// save input and output shapes
			input_2d_shape_ = input_shape;
			output_2d_shape_[0] = output_shape[0];
			
			// map the vectors to their corresponding matrix
			bias_ = Eigen::Map<const Eigen::MatrixXf> (bias.data(), output_shape[0], output_shape[1]);
			weight_ = Eigen::Map<const Eigen::MatrixXf> (weight.data(), input_shape[1], output_shape[1]);
			
			// set the size of thread pool and calculate the spread in thread executions
			thread_pool_.resize(input_shape[0] <= std::thread::hardware_concurrency() ? input_shape[0] : input_shape[0] % std::thread::hardware_concurrency() + 1);
			slope_ = input_shape[0] / thread_pool_.size();
		}

	private:

		std::string activation_;		// activation function name 

		float vW_ = 0.0, vB_ = 0.0;		// momentums of bias and weight
		
		Eigen::MatrixXf weight_, bias_, input_, doutput_, dweight_;		// weight, bias, input, dH, dW matrices
		
		std::function <void (Eigen::MatrixXf &, Eigen::MatrixXf &)> act_func_;		// empty function for activation function
	
};

}

#endif
