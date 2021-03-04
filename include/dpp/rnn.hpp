#ifndef RNN_HPP
#define RNN_HPP

namespace dpp {

class RNN : public Layer {
	public:
		RNN (const int neurons = 1, const float lrate = 0.001, const std::string activation = "sigmoid", const std::string recurrent_activation = "tanh", const float erate = 1e-8) {
			assert (neurons > 0 && lrate > 0.0 && erate > 0.0);
			Eigen::initParallel();
			lrate_ = lrate;
			erate_ = erate;
			output_2d_shape_[1] = neurons;

			act_func_ = std::bind(&RNN::block_tanh, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
			
			recur_act_func_ = std::bind(&RNN::block_sigmoid, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);

			activation_ = activation == "normal" ? activation : activation == "tanh" ? activation : activation == "softmax" ? activation : activation == "relu" ? activation : "sigmoid";
			
			recurrent_activation_ = recurrent_activation == "normal" ? recurrent_activation : recurrent_activation == "tanh" ? recurrent_activation : recurrent_activation == "softmax" ? \
			recurrent_activation : recurrent_activation == "relu" ? recurrent_activation : "sigmoid";
			
			if (activation == "sigmoid") {
				act_func_ = std::bind(&RNN::block_sigmoid, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
			} else if (activation == "softmax") {
			    act_func_ = std::bind(&RNN::block_softmax, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
			} else if (activation == "relu") {
				act_func_ = std::bind(&RNN::block_relu, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
			}

			if (recurrent_activation == "tanh") {
  				recur_act_func_ = std::bind(&RNN::block_tanh, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
			} else if (recurrent_activation == "softmax") {
				recur_act_func_ = std::bind(&RNN::block_softmax, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
			} else if (recurrent_activation == "relu") {
				recur_act_func_ = std::bind(&RNN::block_relu, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
			}
		}
		
		void init(const Eigen::DSizes<ptrdiff_t, 2> input_shape) override {
			assert(input_shape[0] > 0 && input_shape[1] > 0);
			init_flag_ = true;
			limit_ = input_shape[0] - 1;
			input_2d_shape_= input_shape;
			output_2d_shape_[0] = input_shape[0];
			
			bias_h_ = Eigen::MatrixXf::Ones(1, output_2d_shape_[1]);
			bias_igate_ = Eigen::MatrixXf::Ones(1, output_2d_shape_[1]);
			
			igate_ = Eigen::MatrixXf::Zero(input_shape[0], output_2d_shape_[1]);
			digate_ = Eigen::MatrixXf::Zero(input_shape[0], output_2d_shape_[1]);
			
			doutput_ = Eigen::MatrixXf::Zero(input_shape[0], output_2d_shape_[1]);
			output_prev_ = Eigen::MatrixXf::Zero(input_shape[0], output_2d_shape_[1]);
		
			layer_2d_delta_ = Eigen::MatrixXf::Zero(input_shape[0], input_shape[1]);
			layer_2d_output_ = Eigen::MatrixXf::Zero(input_shape[0], output_2d_shape_[1]);

			dweight_igate_ = Eigen::MatrixXf::Zero(input_shape[1], output_2d_shape_[1]);
			du_weight_igate_ = Eigen::MatrixXf::Zero(output_2d_shape_[1], output_2d_shape_[1]); 
			
			u_weight_igate_ = Eigen::MatrixXf::Constant(output_2d_shape_[1], output_2d_shape_[1], 2.0 / output_2d_shape_[1]);
			weight_igate_ = Eigen::MatrixXf::Constant(input_shape[1], output_2d_shape_[1], 2.0 / sqrtf(input_shape[1] * output_2d_shape_[1]));
			weight_h_ = Eigen::MatrixXf::Constant(output_2d_shape_[1], output_2d_shape_[1], 2.0 / sqrtf(output_2d_shape_[1] * output_2d_shape_[1]));
			
			thread_pool_.resize(input_2d_shape_[0] <= std::thread::hardware_concurrency() ? input_2d_shape_[0] : input_2d_shape_[0] % std::thread::hardware_concurrency() + 1);
			slope_ = input_2d_shape_[0] / thread_pool_.size();
		}

		void topology() {
			std::cout << "\n\t----------- RNN Layer -----------\n"
					  << "\n\tParameters:\t(" << weight_igate_.rows() * weight_igate_.cols() * 2 + output_2d_shape_[1] * output_2d_shape_[1] + output_2d_shape_[1] * 2 << ')'
					  << "\n\tInput-dim:\t(" << input_2d_shape_[0] << ", " << input_2d_shape_[1] << ')'
                      << "\n\tOutput-dim:\t(" << output_2d_shape_[0] << ", " << output_2d_shape_[1] << ")\n"
					  << "\n\t--------------------------------\n\n";
		}

		void forward(const Eigen::MatrixXf &X) override {
			assert(X.rows() == input_2d_shape_[0] && X.cols() == input_2d_shape_[1] && init_flag_);
			input_ = X; // save X
			output_prev_.row(0) = layer_2d_output_.row(limit_); // set y (t - 1)

			// loop from 0 to sequence length - 1
			for (size_t w = 0; w < limit_; w ++) {
				// ht = activate(Wx * X + Uh * y (t - 1) + Bh)
				igate_.row(w) = input_.row(w) * weight_igate_ + output_prev_.row(w) * u_weight_igate_ + bias_igate_;
				recur_act_func_(igate_, digate_, w);

				// yt = ractivate(Wh * ht + By) 
				layer_2d_output_.row(w) = igate_.row(w) * weight_h_ + bias_h_;
				act_func_(layer_2d_output_, doutput_ , w);
				
				// set the next y (t - 1)
				output_prev_.row(w + 1) = layer_2d_output_.row(w);
			}

			// perform the last set of calculations
			igate_.row(limit_) = input_.row(limit_) * weight_igate_ + output_prev_.row(limit_) * u_weight_igate_ + bias_igate_;
			recur_act_func_(igate_, digate_, limit_);

			layer_2d_output_.row(limit_) = igate_.row(limit_) * weight_h_ + bias_h_;
			act_func_(layer_2d_output_, doutput_ , limit_);
		}

		void update() {
			input_T_ = input_.transpose();
			output_prev_T_ = output_prev_.transpose();
			weight_igate_T_ = weight_igate_.transpose();
			
			dweight_h_ = doutput_.transpose() * igate_;
			dbias_h_ = doutput_.colwise().sum().array() / output_2d_shape_[0];
			
			doutput_ = doutput_ * weight_h_;
			doutput_ = doutput_.array() * digate_.array();
			dbias_igate_ = doutput_.colwise().sum().array() / output_2d_shape_[0];

			for (size_t w = 0; w < output_2d_shape_[0]; w ++) {
				dweight_igate_ = dweight_igate_ + (input_T_.col(w) * doutput_.row(w));  		
				du_weight_igate_ = du_weight_igate_ + (output_prev_T_.col(w) * doutput_.row(w)); 
				layer_2d_delta_.row(w) = doutput_.row(w) * weight_igate_T_; 
			}	
			
			dweight_igate_ = dweight_igate_.array() / output_2d_shape_[0];
			du_weight_igate_ = du_weight_igate_.array() / output_2d_shape_[0];
			
			vB_h_ = 0.1 * vB_h_ + 0.9 * dbias_h_.array().square().sum();
			vW_h_ = 0.1 * vW_h_ + 0.9 * dweight_h_.array().square().sum();
			vB_igate_ = 0.1 * vB_igate_ + 0.9 * dbias_igate_.array().square().sum();
			vW_igate_ = 0.1 * vW_igate_ + 0.9 * dweight_igate_.array().square().sum();
			vU_igate_ = 0.1 * vU_igate_ + 0.9 * du_weight_igate_.array().square().sum();
			
			bias_h_ = bias_h_.array() - lrate_ / sqrtf(vB_h_ + erate_) * dbias_h_.array();
			weight_h_ = weight_h_.array() - lrate_ / sqrtf(vW_h_ + erate_) * dweight_h_.array();
			bias_igate_ = bias_igate_.array() - lrate_ / sqrtf(vB_igate_ + erate_) * dbias_igate_.array();
			weight_igate_ = weight_igate_.array() - lrate_ / sqrtf(vW_igate_ + erate_) * weight_igate_.array();
			u_weight_igate_ = u_weight_igate_.array() - lrate_ / sqrtf(vU_igate_ + erate_) * u_weight_igate_.array();

			dweight_igate_ = dweight_igate_.array() * 0.0; 
			du_weight_igate_ = du_weight_igate_.array() * 0.0;
		}

		void reset() override {	
			layer_2d_output_.row(limit_) = layer_2d_output_.row(limit_).array() * 0.0;
		}

		void set_delta(const Eigen::MatrixXf &prev_layer_delta) override {
			doutput_ = prev_layer_delta.array() * doutput_.array();
		}

		const std::string name() { return "RNN"; }

		const std::string type() { return "2D"; }

		const float MeanSquaredError(const Eigen::MatrixXf &Y, float &accuracy) override {
			doutput_ = (layer_2d_output_ - Y).array() * doutput_.array();
		   	accuracy = layer_2d_output_.unaryExpr<float(*)(float)>(&roundf).isApprox(Y) ? accuracy + 1 : accuracy;
			return 0.5 * (layer_2d_output_ - Y).array().square().sum();
		}

		const float CrossEntropyError(const Eigen::MatrixXf &Y, float &accuracy) override {
			doutput_ = (layer_2d_output_ - Y);
			float hits = 0.0;

			start_ = 0; end_ = start_ + slope_;
			for (int t = 0; t < thread_pool_.size(); t ++) {
				thread_pool_[t] = std::thread(&RNN::measure2d, this, std::ref(layer_2d_output_), std::ref(Y), std::ref(hits), start_, end_);
				start_ += slope_; end_ += slope_;
			}

			for (int t = 0; t < thread_pool_.size(); t ++) {
				thread_pool_[t].join();
			}

			if (output_2d_shape_[0] % thread_pool_.size() > 0) {
				measure2d(layer_2d_output_, Y, hits, start_, output_2d_shape_[0]);
			}

			accuracy = hits / output_2d_shape_[0] + accuracy;
			return (-Y.array() * layer_2d_output_.array()).log().sum();
		}

		void save(std::ofstream &file) {
			assert(init_flag_);
			Eigen::Map <Eigen::VectorXf> b_igate_vect (bias_igate_.data(), bias_igate_.size()), 
										 w_igate_vect (weight_igate_.data(), weight_igate_.size()), 
										 u_igate_vect (u_weight_igate_.data(), u_weight_igate_.size());

			Eigen::Map <Eigen::VectorXf> bias_vect (bias_h_.data(), bias_h_.size()), 
										 weight_vect (weight_h_.data(), weight_h_.size());	

			file << "\nRNN\n";
			file << "Activation: " << activation_ << '\n';
			file << "Recurrent_Activation: " << recurrent_activation_ << '\n';
			file << "Learning_rate: (" << lrate_ << ")\n";
			file << "Momentum_rate: (" << erate_ << ")\n";
			file << "Input_shape: [" << input_2d_shape_[0] << ',' << input_2d_shape_[1] << "]\n";
			file << "Output_shape: [" << output_2d_shape_[0] << ',' << output_2d_shape_[1] << "]\n";
			file << "Input_gate_bias: [";
			write_vector(b_igate_vect, file);
			file << "Input_gate_weight: [";
			write_vector(w_igate_vect, file);
			file << "Input_gate_uweight: [";
			write_vector(u_igate_vect, file);
			file << "Output_bias: [";
			write_vector(bias_vect, file);
			file << "Output_weight: [";
			write_vector(weight_vect, file);
		}	

		void load(const Eigen::DSizes<ptrdiff_t, 2> &input_shape, 
				  const Eigen::DSizes<ptrdiff_t, 2> &output_shape,
				  const std::vector<float> &bias_igate, const std::vector<float> &weight_igate, const std::vector<float> &u_weight_igate,
				  const std::vector<float> &bias_h, std::vector<float> &weight_h) override {
			
			init_flag_ = true;
			limit_ = input_shape[0] - 1;
			input_2d_shape_ = input_shape;
			output_2d_shape_[0] = output_shape[0];	

			bias_h_ = Eigen::Map<const Eigen::MatrixXf> (bias_h.data(), 1, output_2d_shape_[1]);
			bias_igate_ = Eigen::Map<const Eigen::MatrixXf> (bias_igate.data(), 1, output_2d_shape_[1]);

			weight_h_ = Eigen::Map<const Eigen::MatrixXf> (weight_h.data(), output_2d_shape_[1], output_2d_shape_[1]);
			weight_igate_ = Eigen::Map<const Eigen::MatrixXf> (weight_igate.data(), input_2d_shape_[1], output_2d_shape_[1]);
			u_weight_igate_ = Eigen::Map<const Eigen::MatrixXf> (u_weight_igate.data(), output_2d_shape_[1], output_2d_shape_[1]);
			
			igate_ = Eigen::MatrixXf::Zero(input_shape[0], output_2d_shape_[1]);
			digate_ = Eigen::MatrixXf::Zero(input_shape[0], output_2d_shape_[1]);
			dweight_igate_ = Eigen::MatrixXf::Zero(input_shape[1], output_2d_shape_[1]);
			du_weight_igate_ = Eigen::MatrixXf::Zero(output_2d_shape_[1], output_2d_shape_[1]);

			doutput_ = Eigen::MatrixXf::Zero(input_shape[0], output_2d_shape_[1]);
            layer_2d_delta_ = Eigen::MatrixXf::Zero(input_shape[0], input_shape[1]);
			output_prev_ = Eigen::MatrixXf::Zero(input_shape[0], output_2d_shape_[1]);
            layer_2d_output_ = Eigen::MatrixXf::Zero(input_shape[0], output_2d_shape_[1]);
			
			thread_pool_.resize(input_2d_shape_[0] <= std::thread::hardware_concurrency() ? input_2d_shape_[0] : input_2d_shape_[0] % std::thread::hardware_concurrency() + 1);
			slope_ = input_2d_shape_[0] / thread_pool_.size();
		}
			
	private:
		int limit_; // sequence length - 1
		
		std::string activation_, recurrent_activation_;			// names of activation function

		float vW_igate_, vU_igate_, vB_igate_, vW_h_, vB_h_;	// momentum values for weight and recurrent weight	
		
		// based on Jordan Network 
		Eigen::MatrixXf weight_igate_, u_weight_igate_, bias_igate_, 
						dweight_igate_, du_weight_igate_, dbias_igate_,
						weight_igate_T_,
				        weight_h_, bias_h_, 
						dweight_h_, dbias_h_,
						output_prev_,
						output_prev_T_,	
						igate_, 
						digate_,
						input_,
						input_T_,	
						doutput_,
						doutput_T_;	 

		// empty functions for activation and recurrent activation functions
		std::function <void (Eigen::MatrixXf &, Eigen::MatrixXf &, const size_t &)> act_func_, recur_act_func_;
};

}

#endif
