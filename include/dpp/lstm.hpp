#ifndef LSTM_HPP
#define LSTM_HPP

namespace dpp {

class LSTM : public Layer {

	/*
	 *
	 *	Long Short Term Memory Layer
	 *
	*/

	public:
		LSTM(const int neurons = 1, const float lrate = 0.001, const std::string activation = "tanh", const std::string recurrent_activation = "sigmoid", const float erate = 1e-8) {
			assert(neurons > 0 && lrate > 0.0f && erate > 0.0f);
			/*
             * Save the parameters, and initalize parallelism and set the appropriate activation and recurrent activation functions
            */

			Eigen::initParallel();

			// set learning and optimization rate; save neurons to output shape
			lrate_ = lrate;
			erate_ = erate;
			output_2d_shape_[1] = neurons;
			
			// bind activation function first to block tanh
			act_func_ = std::bind(&LSTM::block_tanh, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);

			// bind recurrent activation function first to block sigmoid
			recur_act_func_ = std::bind(&LSTM::block_sigmoid, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);

			// save the activation name for when saving to a file
			activation_ = activation == "normal" ? activation : activation == "tanh" ? activation : activation == "softmax" ? activation : activation == "relu" ? activation : "sigmoid";

			// save the recurrent activation name for when saving to a file
			recurrent_activation_ = recurrent_activation == "normal" ? recurrent_activation : recurrent_activation == "tanh" ? recurrent_activation : recurrent_activation == "softmax" ? recurrent_activation : recurrent_activation == "relu" ? recurrent_activation : "sigmoid";	

			// make any necessary changes to activation function
			if (activation == "sigmoid") {
				act_func_ = std::bind(&LSTM::block_sigmoid, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
			} else if (activation == "softmax") {
			    act_func_ = std::bind(&LSTM::block_softmax, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
			} else if (activation == "relu") {
				act_func_ = std::bind(&LSTM::block_relu, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
			}

			// make any necessary changes to recurrent activation function
			if (recurrent_activation == "tanh") {
  				recur_act_func_ = std::bind(&LSTM::block_tanh, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
			} else if (recurrent_activation == "softmax") {
				recur_act_func_ = std::bind(&LSTM::block_softmax, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
			} else if (recurrent_activation == "relu") {
				recur_act_func_ = std::bind(&LSTM::block_relu, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
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
			
			// limit of for loop; sequence length - 1
			limit_ = input_shape[0] - 1;

			// save input and output shapes
			input_2d_shape_ = input_shape;
			output_2d_shape_[0] = input_shape[0];
			
			// set shape of layer delta and layer output
			layer_2d_delta_ = Eigen::MatrixXf::Zero(input_shape[0], input_shape[1]);
			layer_2d_output_ = Eigen::MatrixXf::Zero(input_shape[0], output_2d_shape_[1]);

			// initialize bias of LSTM gates
			bias_fgate_ = Eigen::MatrixXf::Ones(1, output_2d_shape_[1]);
			bias_igate_ = Eigen::MatrixXf::Ones(1, output_2d_shape_[1]);
			bias_cgate_ = Eigen::MatrixXf::Ones(1, output_2d_shape_[1]);
			bias_ogate_ = Eigen::MatrixXf::Ones(1, output_2d_shape_[1]);

			// set the shape of each gate output and the layer previous output
			fgate_ = Eigen::MatrixXf::Zero(input_shape[0], output_2d_shape_[1]);
			igate_ = Eigen::MatrixXf::Zero(input_shape[0], output_2d_shape_[1]);
			cgate_ = Eigen::MatrixXf::Zero(input_shape[0], output_2d_shape_[1]);
			ogate_ = Eigen::MatrixXf::Zero(input_shape[0], output_2d_shape_[1]);
			cell_state_ = Eigen::MatrixXf::Zero(input_shape[0], output_2d_shape_[1]);
			output_prev_ = Eigen::MatrixXf::Zero(input_shape[0], output_2d_shape_[1]);

			// set the shape of each gate gradient and previous cell state
			dfgate_ = Eigen::MatrixXf::Zero(input_shape[0], output_2d_shape_[1]);
			digate_ = Eigen::MatrixXf::Zero(input_shape[0], output_2d_shape_[1]);
			dogate_ = Eigen::MatrixXf::Zero(input_shape[0], output_2d_shape_[1]);
			dcgate_ = Eigen::MatrixXf::Zero(input_shape[0], output_2d_shape_[1]);
			dcell_state_ = Eigen::MatrixXf::Zero(input_shape[0], output_2d_shape_[1]);
			cell_state_prev_ = Eigen::MatrixXf::Zero(input_shape[0], output_2d_shape_[1]);

			// set the shape of each gate dW
			dweight_fgate_ = Eigen::MatrixXf::Zero(input_shape[1], output_2d_shape_[1]);
			dweight_igate_ = Eigen::MatrixXf::Zero(input_shape[1], output_2d_shape_[1]);
			dweight_ogate_ = Eigen::MatrixXf::Zero(input_shape[1], output_2d_shape_[1]);
			dweight_cgate_ = Eigen::MatrixXf::Zero(input_shape[1], output_2d_shape_[1]);

			// set the shape of each gate dU
			du_weight_fgate_ = Eigen::MatrixXf::Zero(output_2d_shape_[1], output_2d_shape_[1]);
			du_weight_igate_ = Eigen::MatrixXf::Zero(output_2d_shape_[1], output_2d_shape_[1]);
			du_weight_ogate_ = Eigen::MatrixXf::Zero(output_2d_shape_[1], output_2d_shape_[1]);
			du_weight_cgate_ = Eigen::MatrixXf::Zero(output_2d_shape_[1], output_2d_shape_[1]);

			// initialize the recurrent weight of each LSTM gate
			u_weight_fgate_ = Eigen::MatrixXf::Constant(output_2d_shape_[1], output_2d_shape_[1], 2.0f / sqrtf(output_2d_shape_[1] * output_2d_shape_[1]));
			u_weight_igate_ = Eigen::MatrixXf::Constant(output_2d_shape_[1], output_2d_shape_[1], 2.0f / sqrtf(output_2d_shape_[1] * output_2d_shape_[1]));
			u_weight_ogate_ = Eigen::MatrixXf::Constant(output_2d_shape_[1], output_2d_shape_[1], 2.0f / sqrtf(output_2d_shape_[1] * output_2d_shape_[1]));
			u_weight_cgate_ = Eigen::MatrixXf::Constant(output_2d_shape_[1], output_2d_shape_[1], 2.0f / sqrtf(output_2d_shape_[1] * output_2d_shape_[1]));

			// initialize the weight of each LSTM gate
			weight_fgate_ = Eigen::MatrixXf::Constant(input_shape[1], output_2d_shape_[1], 2.0f / sqrtf(input_shape[1] * output_2d_shape_[1]));
			weight_igate_ = Eigen::MatrixXf::Constant(input_shape[1], output_2d_shape_[1], 2.0f / sqrtf(input_shape[1] * output_2d_shape_[1]));
			weight_ogate_ = Eigen::MatrixXf::Constant(input_shape[1], output_2d_shape_[1], 2.0f / sqrtf(input_shape[1] * output_2d_shape_[1]));
			weight_cgate_ = Eigen::MatrixXf::Constant(input_shape[1], output_2d_shape_[1], 2.0f / sqrtf(input_shape[1] * output_2d_shape_[1]));
		
			// select the appropriate size of thread pool and calculate the spread in thread executions	
			thread_pool_.resize(input_shape[0] <= std::thread::hardware_concurrency() ? input_shape[0] : input_shape[0] % std::thread::hardware_concurrency() + 1);
			slope_ = input_shape[0] / thread_pool_.size();

		}

		void topology() {
			std::cout << "\n\t------------- LSTM Layer ---------------\n"
					  << "\n\tParameters:\t(" << output_2d_shape_[1] * output_2d_shape_[1] * 4 + output_2d_shape_[1] * output_2d_shape_[0] * 4 + output_2d_shape_[1] * 4 << ')'
					  << "\n\tInput-dim:\t(" << input_2d_shape_[0] << ", " << input_2d_shape_[1] << ')'
                      << "\n\tOutput-dim:\t(" << output_2d_shape_[0] << ", " << output_2d_shape_[1] << ")\n"
					  << "\n\t----------------------------------------\n\n";
		}

		void forward(const Eigen::MatrixXf &X) override {
			assert(X.rows() == input_2d_shape_[0] && X.cols() == input_2d_shape_[1] && init_flag_);
			/* *
			   * Perform H Calculation sequentially
               * Save X
			   * Set previous H and precious cell state
               * Loop through sequence
               * Calculate output of each gate
			   * Calculate Cell state
			   * Calculate H
             * */

			input_ = X;
			output_prev_.row(0) = layer_2d_output_.row(limit_);
			cell_state_prev_.row(0) = cell_state_.row(limit_);
			
			for (size_t w = 0; w < limit_; w ++) {
				// calculate Z of each gate
				fgate_.row(w) = output_prev_.row(w) * u_weight_fgate_ + input_.row(w) * weight_fgate_ + bias_fgate_; 
				igate_.row(w) = output_prev_.row(w) * u_weight_igate_ + input_.row(w) * weight_igate_ + bias_igate_;
				ogate_.row(w) = output_prev_.row(w) * u_weight_ogate_ + input_.row(w) * weight_ogate_ + bias_ogate_;
				cgate_.row(w) = output_prev_.row(w) * u_weight_cgate_ + input_.row(w) * weight_cgate_ + bias_cgate_;
				
				// activate Z of LSTM gates 
				act_func_(cgate_, dcgate_, w);				// tanh
				recur_act_func_(fgate_, dfgate_, w);		// sigmoid
				recur_act_func_(igate_, digate_, w);		// sigmoid
				recur_act_func_(ogate_, dogate_, w);		// sigmoid
				
				// calculate cell state
				cell_state_.row(w) = fgate_.row(w).array() * cell_state_prev_.row(w).array() + igate_.row(w).array() * cgate_.row(w).array();
				
				// activate cell state
				act_func_(cell_state_, dcell_state_, w);	// tanh

				// calculate H
				layer_2d_output_.row(w) = ogate_.row(w).array() * cell_state_.row(w).array();

				// set previous H and previous cell state
				output_prev_.row(w + 1) = layer_2d_output_.row(w);	
				cell_state_prev_.row(w + 1) = cell_state_.row(w);
			}
			
			// calculate the last set of calculations
			fgate_.row(limit_) = output_prev_.row(limit_) * u_weight_fgate_ + input_.row(limit_) * weight_fgate_ + bias_fgate_;
            igate_.row(limit_) = output_prev_.row(limit_) * u_weight_igate_ + input_.row(limit_) * weight_igate_ + bias_igate_;
            ogate_.row(limit_) = output_prev_.row(limit_) * u_weight_ogate_ + input_.row(limit_) * weight_ogate_ + bias_ogate_;
            cgate_.row(limit_) = output_prev_.row(limit_) * u_weight_cgate_ + input_.row(limit_) * weight_cgate_ + bias_cgate_;

            act_func_(cgate_, dcgate_, limit_);
            recur_act_func_(fgate_, dfgate_, limit_);
            recur_act_func_(igate_, digate_, limit_);
			recur_act_func_(ogate_, dogate_, limit_);

			cell_state_.row(limit_) = fgate_.row(limit_).array() * cell_state_prev_.row(limit_).array() + igate_.row(limit_).array() * cgate_.row(limit_).array();
            act_func_(cell_state_, dcell_state_, limit_);

            layer_2d_output_.row(limit_) = ogate_.row(limit_).array() * cell_state_.row(limit_).array();
		}

		void update() {
			// calculate output gate gradient
			dogate_ = doutput_.array() * cell_state_.array() * dogate_.array();  

			// set new dH for calculations of other three gates' gradient
			doutput_ = doutput_.array() * ogate_.array() * dcell_state_.array();

			// calculate forget gate's gradient
			dfgate_ = doutput_.array() * cell_state_prev_.array() * dfgate_.array();

			// calculate input gate's gradient
			digate_ = doutput_.array() * cgate_.array() * digate_.array();

			// calculate memory gate's gradient
			dcgate_ = doutput_.array() * igate_.array() * dcgate_.array();
			
			// transpose input and save it to another matrix
			input_T_ = input_.transpose();

			// transpose H_prev and save it to another matrix
			output_prev_T_ = output_prev_.transpose();

			// transpose all of the gates' weight
			weight_fgate_T_ = weight_fgate_.transpose();
			weight_igate_T_ = weight_igate_.transpose();
			weight_cgate_T_ = weight_cgate_.transpose();
			weight_ogate_T_ = weight_ogate_.transpose();
			
			for (size_t w = 0; w < output_2d_shape_[0]; w ++) {

				// summation of each gate weight gradient per word in sequence
				dweight_fgate_ = dweight_fgate_ + input_T_.col(w) * dfgate_.row(w);
				dweight_igate_ = dweight_igate_ + input_T_.col(w) * digate_.row(w);
				dweight_ogate_ = dweight_ogate_ + input_T_.col(w) * dogate_.row(w);
				dweight_cgate_ = dweight_cgate_ + input_T_.col(w) * dcgate_.row(w);
						
				// summation of each gate recurrent weight gradient per word in sequence
				du_weight_fgate_ = du_weight_fgate_ + output_prev_T_.col(w) * dfgate_.row(w);
				du_weight_igate_ = du_weight_igate_ + output_prev_T_.col(w) * digate_.row(w);
				du_weight_ogate_ = du_weight_ogate_ + output_prev_T_.col(w) * dogate_.row(w);
				du_weight_cgate_ = du_weight_cgate_ + output_prev_T_.col(w) * dcgate_.row(w);
				
				// calculate layer gradient
				layer_2d_delta_.row(w) = (dfgate_.row(w) * weight_fgate_T_) + (digate_.row(w) * weight_igate_T_) + (dogate_.row(w) * weight_ogate_T_) + (dcgate_.row(w) * weight_cgate_T_);
			}
			
			// average the weight-gradients 
			dweight_fgate_ = dweight_fgate_.array() / output_2d_shape_[0];
			dweight_igate_ = dweight_igate_.array() / output_2d_shape_[0];
			dweight_ogate_ = dweight_ogate_.array() / output_2d_shape_[0];
			dweight_cgate_ = dweight_cgate_.array() / output_2d_shape_[0];

			// average the recurrent weight gradients
			du_weight_fgate_ = du_weight_fgate_.array() / output_2d_shape_[0];
			du_weight_igate_ = du_weight_igate_.array() / output_2d_shape_[0];
			du_weight_ogate_ = du_weight_ogate_.array() / output_2d_shape_[0];
			du_weight_cgate_ = du_weight_cgate_.array() / output_2d_shape_[0];
			
			// average the each gate's output gradient by summing over their rows (e.g., 100 x 100 -> 1 x 100)
			dbias_fgate_ = dfgate_.colwise().sum().array() / output_2d_shape_[0];
			dbias_igate_ = digate_.colwise().sum().array() / output_2d_shape_[0];
			dbias_ogate_ = dogate_.colwise().sum().array() / output_2d_shape_[0];
			dbias_cgate_ = dcgate_.colwise().sum().array() / output_2d_shape_[0];
			
			// calculate each gate bias momentum
			vB_fgate_ = 0.1 * vB_fgate_ + 0.9 * dbias_fgate_.array().square().sum();
			vB_igate_ = 0.1 * vB_igate_ + 0.9 * dbias_igate_.array().square().sum();
			vB_ogate_ = 0.1 * vB_ogate_ + 0.9 * dbias_ogate_.array().square().sum();	
            vB_cgate_ = 0.1 * vB_cgate_ + 0.9 * dbias_cgate_.array().square().sum();

			// calculate each gate's weight momentum
			vW_fgate_ = 0.1 * vW_fgate_ + 0.9 * dweight_fgate_.array().square().sum();
			vW_igate_ = 0.1 * vW_igate_ + 0.9 * dweight_igate_.array().square().sum();
			vW_ogate_ = 0.1 * vW_ogate_ + 0.9 * dweight_ogate_.array().square().sum();
			vW_cgate_ = 0.1 * vW_cgate_ + 0.9 * dweight_cgate_.array().square().sum();

			// calculate each gate's recurrent weight momentum
			vU_fgate_ = 0.1 * vU_fgate_ + 0.9 * du_weight_fgate_.array().square().sum();
            vU_igate_ = 0.1 * vU_igate_ + 0.9 * du_weight_igate_.array().square().sum();
            vU_ogate_ = 0.1 * vU_ogate_ + 0.9 * du_weight_ogate_.array().square().sum();
            vU_cgate_ = 0.1 * vU_cgate_ + 0.9 * du_weight_cgate_.array().square().sum();
			
			// change each gate's weight
			weight_fgate_ = weight_fgate_.array() - lrate_ / sqrtf(vW_fgate_ + erate_) * dweight_fgate_.array();
			weight_igate_ = weight_igate_.array() - lrate_ / sqrtf(vW_igate_ + erate_) * dweight_igate_.array();
			weight_ogate_ = weight_ogate_.array() - lrate_ / sqrtf(vW_ogate_ + erate_) * dweight_ogate_.array();
			weight_cgate_ = weight_cgate_.array() - lrate_ / sqrtf(vW_cgate_ + erate_) * dweight_cgate_.array();
			
			// change each gate's recurrent weight
			u_weight_fgate_ = u_weight_fgate_.array() - lrate_ / sqrtf(vU_fgate_ + erate_) * du_weight_fgate_.array();
			u_weight_igate_ = u_weight_igate_.array() - lrate_ / sqrtf(vU_igate_ + erate_) * du_weight_igate_.array();
			u_weight_ogate_ = u_weight_ogate_.array() - lrate_ / sqrtf(vU_ogate_ + erate_) * du_weight_ogate_.array();
			u_weight_cgate_ = u_weight_cgate_.array() - lrate_ / sqrtf(vU_cgate_ + erate_) * du_weight_cgate_.array();
			
			// change each gate's bias 
			bias_fgate_ = bias_fgate_.array() - lrate_ / sqrtf(vB_fgate_ + erate_) * dbias_fgate_.array();
			bias_igate_ = bias_igate_.array() - lrate_ / sqrtf(vB_igate_ + erate_) * dbias_igate_.array();
			bias_ogate_ = bias_ogate_.array() - lrate_ / sqrtf(vB_ogate_ + erate_) * dbias_ogate_.array();
			bias_cgate_ = bias_cgate_.array() - lrate_ / sqrtf(vB_cgate_ + erate_) * dbias_cgate_.array();
			
			// reset all gradients to 0
			dweight_fgate_ = dweight_fgate_.array() * 0; dweight_ogate_ = dweight_ogate_.array() * 0; 
			dweight_igate_ = dweight_igate_.array() * 0; dweight_cgate_ = dweight_cgate_.array() * 0; 
			du_weight_fgate_ = du_weight_fgate_.array() * 0; du_weight_ogate_ = du_weight_ogate_.array() * 0;	
			du_weight_igate_ = du_weight_igate_.array() * 0; du_weight_cgate_ = du_weight_cgate_.array() * 0;
		}

		void reset() override {
			cell_state_.row(limit_) = cell_state_.row(limit_).array() * 0.0; 
			layer_2d_output_.row(limit_) = layer_2d_output_.row(limit_).array() * 0.0; 
		}

		void set_delta(const Eigen::MatrixXf &prev_layer_delta) override {
			// dH = dJ (l + 1) as LSTM does not have an appropriate general dH 
			doutput_ = prev_layer_delta; 
		}

		const std::string name() { return "LSTM"; }

		const std::string type() { return "2D"; }

		const float MeanSquaredError(const Eigen::MatrixXf &Y, float &accuracy) override {
			// dH = H - Y
			doutput_ = layer_2d_output_ - Y;

		   	accuracy = layer_2d_output_.unaryExpr<float(*)(float)>(&roundf).isApprox(Y) ? accuracy + 1 : accuracy;
			
			return 0.5 * doutput_.array().square().sum();
		}
		
		const float CrossEntropyError(const Eigen::MatrixXf &Y, float &accuracy) override {
			/*
             * dJ = (-Y / H)
             * dH = dJ as dH is empty
            */	

			// matches of each row
			float hits = 0.0;

			// set new dH
			doutput_ = -Y.array() / layer_2d_output_.array();
			
			// initalize start and end index of thread
			start_ = 0; end_ = slope_;

			// measure accuracy of H concurrently
			for (int t = 0; t < thread_pool_.size(); t ++) {
				thread_pool_[t] = std::thread(&LSTM::measure2d, this, std::ref(layer_2d_output_), std::ref(Y), std::ref(hits), start_, end_);
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
			
			// convert the layer's parameters into vector
			Eigen::Map<Eigen::VectorXf> w_fgate_vect (weight_fgate_.data(), weight_fgate_.size()), u_fgate_vect (u_weight_fgate_.data(), u_weight_fgate_.size()),
										w_igate_vect (weight_igate_.data(), weight_igate_.size()), u_igate_vect (u_weight_igate_.data(), u_weight_igate_.size()),
										w_ogate_vect (weight_ogate_.data(), weight_ogate_.size()), u_ogate_vect (u_weight_ogate_.data(), u_weight_ogate_.size()),
										w_cgate_vect (weight_cgate_.data(), weight_cgate_.size()), u_cgate_vect (u_weight_cgate_.data(), u_weight_cgate_.size()),
									   	b_fgate_vect (bias_fgate_.data(), bias_fgate_.size()),
										b_igate_vect (bias_igate_.data(), bias_igate_.size()),
									   	b_ogate_vect (bias_ogate_.data(), bias_ogate_.size()),
									   	b_cgate_vect (bias_cgate_.data(), bias_cgate_.size());

			// write to file
			file << "\nLSTM\n";
			file << "Activation: " << activation_ << '\n';	
			file << "Recurrent_Activation: " << recurrent_activation_ << '\n';
			file << "Learning_rate: (" << lrate_ << ")\n";
            file << "Momentum_rate: (" << erate_ << ")\n";
			file << "Input_shape: [" << input_2d_shape_[0] << ',' << input_2d_shape_[1] << "]\n";
			file << "Output_shape: [" << output_2d_shape_[0] << ',' << output_2d_shape_[1] << "]\n";
			file << "Forget_gate_bias: [";
			write_vector(b_fgate_vect, file);
			file << "Forget_gate_weight: [";
			write_vector(w_fgate_vect, file);	
			file << "Forget_gate_uweight: [";
			write_vector(u_fgate_vect, file);

			file << "Input_gate_bias: [";
			write_vector(b_igate_vect, file);
			file << "Input_gate_weight: [";
			write_vector(w_igate_vect, file);	
			file << "Input_gate_uweight: [";
			write_vector(u_igate_vect, file);

			file << "Output_gate_bias: [";
			write_vector(b_ogate_vect, file);
			file << "Output_gate_weight: [";
			write_vector(w_ogate_vect, file);	
			file << "Output_gate_uweight: [";
			write_vector(u_ogate_vect, file);

			file << "Memory_gate_bias: [";
			write_vector(b_cgate_vect, file);
			file << "Memory_gate_weight: [";
			write_vector(w_cgate_vect, file);	
			file << "Memory_gate_uweight: [";
			write_vector(u_cgate_vect, file);
		}

	    void load(const Eigen::DSizes<ptrdiff_t, 2> &input_shape,
				  const Eigen::DSizes<ptrdiff_t, 2> &output_shape,
				  const std::vector<float> &bias_fgate, const std::vector<float> &weight_fgate, const std::vector<float> &u_weight_fgate,
				  const std::vector<float> &bias_igate, const std::vector<float> &weight_igate, const std::vector<float> &u_weight_igate,
				  const std::vector<float> &bias_ogate, const std::vector<float> &weight_ogate, const std::vector<float> &u_weight_ogate,
				  const std::vector<float> &bias_cgate, const std::vector<float> &weight_cgate, const std::vector<float> &u_weight_cgate) {
			
			// set the parameters to metadata inside of the layer
			init_flag_ = true;			

			// set limit or sequence length - 1
			limit_ = input_shape[0] - 1;

			// save input and output shapes
			input_2d_shape_ = input_shape;
			output_2d_shape_[0] = input_shape[0];

			// set the shape of layer delta, and layer output	
			layer_2d_delta_ = Eigen::MatrixXf::Zero(input_shape[0], input_shape[1]);
			layer_2d_output_ = Eigen::MatrixXf::Zero(input_shape[0], output_2d_shape_[1]);
			
			// set the shape of each gate output and the layer previous output
			fgate_ = Eigen::MatrixXf::Zero(input_shape[0], output_2d_shape_[1]);
			igate_ = Eigen::MatrixXf::Zero(input_shape[0], output_2d_shape_[1]);
			cgate_ = Eigen::MatrixXf::Zero(input_shape[0], output_2d_shape_[1]);
			ogate_ = Eigen::MatrixXf::Zero(input_shape[0], output_2d_shape_[1]);
			cell_state_ = Eigen::MatrixXf::Zero(input_shape[0], output_2d_shape_[1]);
			output_prev_ = Eigen::MatrixXf::Zero(input_shape[0], output_2d_shape_[1]);

			// set the shape of each gate gradient and previous cell state
			dfgate_ = Eigen::MatrixXf::Zero(input_shape[0], output_2d_shape_[1]);
			digate_ = Eigen::MatrixXf::Zero(input_shape[0], output_2d_shape_[1]);
			dogate_ = Eigen::MatrixXf::Zero(input_shape[0], output_2d_shape_[1]);
			dcgate_ = Eigen::MatrixXf::Zero(input_shape[0], output_2d_shape_[1]);
			dcell_state_ = Eigen::MatrixXf::Zero(input_shape[0], output_2d_shape_[1]);
			cell_state_prev_ = Eigen::MatrixXf::Zero(input_shape[0], output_2d_shape_[1]);
			
			// set the shape of each gate dW
			dweight_fgate_ = Eigen::MatrixXf::Zero(input_shape[1], output_2d_shape_[1]);
			dweight_igate_ = Eigen::MatrixXf::Zero(input_shape[1], output_2d_shape_[1]);
			dweight_ogate_ = Eigen::MatrixXf::Zero(input_shape[1], output_2d_shape_[1]);
			dweight_cgate_ = Eigen::MatrixXf::Zero(input_shape[1], output_2d_shape_[1]);

			// set the shape of each gate dU
			du_weight_fgate_ = Eigen::MatrixXf::Zero(output_2d_shape_[1], output_2d_shape_[1]);
			du_weight_igate_ = Eigen::MatrixXf::Zero(output_2d_shape_[1], output_2d_shape_[1]);
			du_weight_ogate_ = Eigen::MatrixXf::Zero(output_2d_shape_[1], output_2d_shape_[1]);
			du_weight_cgate_ = Eigen::MatrixXf::Zero(output_2d_shape_[1], output_2d_shape_[1]);

			// map bias vectors to their corresponding matrix
			bias_fgate_ = Eigen::Map<const Eigen::MatrixXf> (bias_fgate.data(), 1, output_shape[1]);
			bias_igate_ = Eigen::Map<const Eigen::MatrixXf> (bias_igate.data(), 1, output_shape[1]);
			bias_ogate_ = Eigen::Map<const Eigen::MatrixXf> (bias_ogate.data(), 1, output_shape[1]);
			bias_cgate_ = Eigen::Map<const Eigen::MatrixXf> (bias_cgate.data(), 1, output_shape[1]);

			// map weight vectors to their corresponding matrix
			weight_fgate_ = Eigen::Map<const Eigen::MatrixXf> (weight_fgate.data(), input_shape[1], output_shape[1]);
			weight_igate_ = Eigen::Map<const Eigen::MatrixXf> (weight_igate.data(), input_shape[1], output_shape[1]);
			weight_ogate_ = Eigen::Map<const Eigen::MatrixXf> (weight_ogate.data(), input_shape[1], output_shape[1]);
			weight_cgate_ = Eigen::Map<const Eigen::MatrixXf> (weight_cgate.data(), input_shape[1], output_shape[1]);

			// map recurrent weight vectors to their corresponding matrix
			u_weight_fgate_ = Eigen::Map<const Eigen::MatrixXf> (u_weight_fgate.data(), output_shape[1], output_shape[1]);
			u_weight_igate_ = Eigen::Map<const Eigen::MatrixXf> (u_weight_igate.data(), output_shape[1], output_shape[1]);
			u_weight_ogate_ = Eigen::Map<const Eigen::MatrixXf> (u_weight_ogate.data(), output_shape[1], output_shape[1]);
			u_weight_cgate_ = Eigen::Map<const Eigen::MatrixXf> (u_weight_cgate.data(), output_shape[1], output_shape[1]);
			
			// set the size of thread pool and calculate the spread in thread executions
			thread_pool_.resize(input_shape[0] <= std::thread::hardware_concurrency() ? input_shape[0] : input_shape[0] % std::thread::hardware_concurrency() + 1);
			slope_ = input_shape[0] / thread_pool_.size();
		}

	private:
		int limit_;				// sequence length - 1
		
		std::string activation_, recurrent_activation_;			 // names of activation function

		// momentum values for weight, recurrent weight, and bias of LSTM gates
		float vW_fgate_ = 0.0f, vU_fgate_ = 0.0f, vB_fgate_ = 0.0f, vW_igate_ = 0.0f, vU_igate_ = 0.0f, vB_igate_ = 0.0f,
			  vW_cgate_ = 0.0f, vU_cgate_ = 0.0f, vB_cgate_ = 0.0f, vW_ogate_ = 0.0f, vU_ogate_ = 0.0f, vB_ogate_ = 0.0f;

		// bias gradient matrix of each gate
		Eigen::MatrixXf dbias_fgate_, dbias_cgate_, dbias_ogate_, dbias_igate_;

		// input_transpose, H_prev_transpose, and transposed weight of each gate matrices
		Eigen::MatrixXf input_T_, output_prev_T_, weight_fgate_T_, weight_igate_T_, weight_cgate_T_, weight_ogate_T_;

		// input, dH, H_prev, cell state, cell state previous, cell state gradient, forget gate, input gate, memory gate, output gate, and each gate gradient
		Eigen::MatrixXf input_, doutput_, output_prev_, cell_state_, cell_state_prev_, dcell_state_, fgate_, igate_, cgate_, ogate_, dfgate_, digate_, dcgate_, dogate_;

		// dW and dU of each gate
		Eigen::MatrixXf dweight_fgate_, du_weight_fgate_, dweight_igate_, du_weight_igate_, dweight_cgate_, du_weight_cgate_, dweight_ogate_, du_weight_ogate_;

		// weight, recurrent weight, and bias of each gate
		Eigen::MatrixXf weight_fgate_, u_weight_fgate_, bias_fgate_, weight_igate_, u_weight_igate_, bias_igate_,
					    weight_cgate_, u_weight_cgate_, bias_cgate_, weight_ogate_, u_weight_ogate_, bias_ogate_;

		// empty functions for activation and recurrent activation functions
		std::function <void (Eigen::MatrixXf &, Eigen::MatrixXf &, const size_t &)> act_func_, recur_act_func_;
};

}

#endif
