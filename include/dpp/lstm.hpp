#ifndef LSTM_HPP
#define LSTM_HPP

#include "layer.hpp" 

namespace dpp {

class LSTM : public Layer {

	/*
	 *	Long Short Term Memory Layer
	*/

	public:
		LSTM(const int neurons = 1, const std::string activation = "tanh", const std::string recurrent_activation = "sigmoid", const float lrate = 0.001, const float erate = 1e-8) {
			assert(neurons > 0 && lrate > 0.0f && erate > 0.0f);
			/*
             * Save the parameters, and initalize parallelism and set the appropriate activation and recurrent activation functions
            */

			Eigen::initParallel();

			// set learning and optimization rate; save neurons to output shape
			lrate_ = lrate;
			erate_ = erate;
			output_2d_shape_[1] = neurons;

			// set size of vectors
			bias_.resize(4);
			gate_.resize(4);
			state_.resize(4);
			dgate_.resize(4);
			dbias_.resize(4);
			x_weight_.resize(4);
			h_weight_.resize(4);
			dx_weight_.resize(4);
			dh_weight_.resize(4);

			// save the activation name for when saving to a file
			activation_ = activation == "normal" ? activation : activation == "softsign" ? activation : activation == "softmax" ? activation : activation == "relu" ? \ 
						  activation : activation == "sigmoid" ? activation : "tanh";

			// save the recurrent activation name for when saving to a file
			recurrent_activation_ = recurrent_activation == "normal" ? recurrent_activation : recurrent_activation == "tanh" ? recurrent_activation : recurrent_activation == "softmax" ? \
								    recurrent_activation : recurrent_activation == "relu" ? recurrent_activation : recurrent_activation == "softsign" ? recurrent_activation : "sigmoid";	

			// make any necessary changes to the activation functions
			set_activation(cell_act_fun_, activation_);

			set_activation(gate_act_fun_, recurrent_activation_);
		}

		void init(const Eigen::DSizes<ptrdiff_t, 2> input_shape) override {
			assert(input_shape[0] > 0 && input_shape[1] > 0);
            /*
             * Initialize the parameters of the layer and save the input shape
             */

            // set initialized flag
            init_flag_ = true;
			
			// sequence length - 1
			limit_ = input_shape[0] - 1;

			// save input and output shapes
            input_2d_shape_ = input_shape;
            output_2d_shape_[0] = input_2d_shape_[0];

            // set shape of layer delta and layer output
			layer_2d_delta_ = Eigen::MatrixXf::Zero(input_2d_shape_[0], input_2d_shape_[1]);
			layer_2d_output_ = Eigen::MatrixXf::Zero(input_2d_shape_[0], output_2d_shape_[1]);
			
			// set the shape of d(activated(C(t)) and other states (C(t), activated C(t), C(t - 1), H(t - 1)) 
			dh_cell_state_ = Eigen::MatrixXf::Zero(input_2d_shape_[0], output_2d_shape_[1]);
			std::for_each(state_.begin(), state_.end(), [this](auto &mat) { mat = Eigen::MatrixXf::Zero(input_2d_shape_[0], output_2d_shape_[1]); });

			// set the shape of each gate and their gradient 
			std::for_each(gate_.begin(), gate_.end(), [this](auto &mat) { mat = Eigen::MatrixXf::Zero(input_2d_shape_[0], output_2d_shape_[1]); });
			std::for_each(dgate_.begin(), dgate_.end(), [this](auto &mat) { mat = Eigen::MatrixXf::Zero(input_2d_shape_[0], output_2d_shape_[1]); });

			// initialize biases and their gradient
			std::for_each(bias_.begin(), bias_.end(), [this](auto &mat) { mat = Eigen::MatrixXf::Ones(1, output_2d_shape_[1]); });
			std::for_each(dbias_.begin(), dbias_.end(), [this](auto &mat) { mat = Eigen::MatrixXf::Zero(1, output_2d_shape_[1]); });


			// initialize X-weights and their gradient 
			std::for_each(x_weight_.begin(), x_weight_.end(), [this](auto &mat) { mat = Eigen::MatrixXf::Constant(input_2d_shape_[1], output_2d_shape_[1], 
																						2.0f / sqrtf(input_2d_shape_[1] * output_2d_shape_[1])); }); 
			
			std::for_each(dx_weight_.begin(), dx_weight_.end(), [this](auto &mat) { mat = Eigen::MatrixXf::Zero(input_2d_shape_[1], output_2d_shape_[1]); }); 

			// initialize H-weights and their gradient
			std::for_each(h_weight_.begin(), h_weight_.end(), [this](auto &mat) { mat = Eigen::MatrixXf::Constant(output_2d_shape_[1], output_2d_shape_[1], 
																					   2.0f / output_2d_shape_[1]); });	

			std::for_each(dh_weight_.begin(), dh_weight_.end(), [this](auto &mat) { mat = Eigen::MatrixXf::Zero(output_2d_shape_[1], output_2d_shape_[1]); });
			
			// select the appropriate size of thread pool and calculate the spread in thread executions	
			thread_pool_.resize(input_2d_shape_[0] <= std::thread::hardware_concurrency() ? input_2d_shape_[0] : input_2d_shape_[0] % std::thread::hardware_concurrency() + 1);

			slope_ = input_2d_shape_[0] / thread_pool_.size();
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
               * Save X
			   * Set previous H and precious cell state
               * Loop through sequence
               * Calculate output of each gate
			   * Calculate Cell state
			   * Calculate H
             * */

			input_ = X;			// store X
			
			// store C(t - 1)	
			state_[2].row(0) = state_[0].row(limit_);

			// store H(t - 1)
			state_[3].row(0) = layer_2d_output_.row(limit_);

			// loop through sequence
			for (int w = 0; w < limit_; w ++) {
				// calculate Z of each gate
				gate_[0].row(w) = input_.row(w) * x_weight_[0] + state_[3].row(w) * h_weight_[0] + bias_[0];
				gate_[1].row(w) = input_.row(w) * x_weight_[1] + state_[3].row(w) * h_weight_[1] + bias_[1];
				gate_[2].row(w) = input_.row(w) * x_weight_[2] + state_[3].row(w) * h_weight_[2] + bias_[2];
				gate_[3].row(w) = input_.row(w) * x_weight_[3] + state_[3].row(w) * h_weight_[3] + bias_[3];
				
				// activate LSTM gates
				gate_act_fun_(gate_[0], dgate_[0], w);			// forget gate 
				gate_act_fun_(gate_[1], dgate_[1], w);			// input gate 
				gate_act_fun_(gate_[2], dgate_[2], w);			// output gate 
				cell_act_fun_(gate_[3], dgate_[3], w);			// mem. gate 

				// calculate cell state : ft * c(t - 1) + it * gt  
				state_[1].row(w) = gate_[0].row(w).cwiseProduct(state_[2].row(w)) + gate_[1].row(w).cwiseProduct(gate_[3].row(w));

				// save c(t)
				state_[0].row(w) = state_[1].row(w);

				// activate c(t)
				cell_act_fun_(state_[1], dh_cell_state_, w);

				// calculate h(t)
				layer_2d_output_.row(w) = gate_[2].row(w).cwiseProduct(state_[1].row(w));
				
				// set next h (t - 1) and c (t - 1)
				state_[2].row(w + 1) = state_[0].row(w);
				state_[3].row(w + 1) = layer_2d_output_.row(w);
			}

			// calculate Z of each gate
			gate_[0].row(limit_) = input_.row(limit_) * x_weight_[0] + state_[3].row(limit_) * h_weight_[0] + bias_[0];
			gate_[1].row(limit_) = input_.row(limit_) * x_weight_[1] + state_[3].row(limit_) * h_weight_[1] + bias_[1];
			gate_[2].row(limit_) = input_.row(limit_) * x_weight_[2] + state_[3].row(limit_) * h_weight_[2] + bias_[2];
			gate_[3].row(limit_) = input_.row(limit_) * x_weight_[3] + state_[3].row(limit_) * h_weight_[3] + bias_[3];
			
			// activate LSTM gates
			gate_act_fun_(gate_[0], dgate_[0], limit_);			// forget gate 
			gate_act_fun_(gate_[1], dgate_[1], limit_);			// input gate 
			gate_act_fun_(gate_[2], dgate_[2], limit_);			// output gate 
			cell_act_fun_(gate_[3], dgate_[3], limit_);			// mem. gate 

			// calculate cell state : ft * c(t - 1) + it * gt  
			state_[1].row(limit_) = gate_[0].row(limit_).cwiseProduct(state_[2].row(limit_)) + gate_[1].row(limit_).cwiseProduct(gate_[3].row(limit_));

			// save c(t)
			state_[0].row(limit_) = state_[1].row(limit_);

			// activate c(t)
			cell_act_fun_(state_[1], dh_cell_state_, limit_);

			// calculate h(t)
			layer_2d_output_.row(limit_) = gate_[2].row(limit_).cwiseProduct(state_[1].row(limit_));
		}
		
		void update() {
			// calculate dH/dZ_gate
			// calculate dCt/dZgate
			// create dhp and dcp 
			// start backward and perform truncated backpropagation through time
			
			// gradients for dC(t - 1)
			dcig_ = dgate_[1].cwiseProduct(gate_[3]);
			dcmg_ = dgate_[3].cwiseProduct(gate_[1]);	
			dcfg_ = dgate_[0].cwiseProduct(state_[2]);

			// memory gate
			dgate_[3] = gate_[2].cwiseProduct(dh_cell_state_).cwiseProduct(dgate_[3]).cwiseProduct(gate_[1]);

			// input gate
			dgate_[1] = gate_[2].cwiseProduct(dh_cell_state_).cwiseProduct(dgate_[1]).cwiseProduct(gate_[3]);
			
			// forget gate
			dgate_[0] = gate_[2].cwiseProduct(dh_cell_state_).cwiseProduct(dgate_[0]).cwiseProduct(state_[2]);	

			// output gate
			dgate_[2] = dgate_[2].cwiseProduct(state_[1]);

			// dJ/dGate
			dhfg_ = doutput_.cwiseProduct(dgate_[0]);
			dhig_ = doutput_.cwiseProduct(dgate_[1]);
			dhog_ = doutput_.cwiseProduct(dgate_[2]);
			dhmg_ = doutput_.cwiseProduct(dgate_[3]);

			// transpose input, cp, and weight matrices in place
			input_.transposeInPlace();
			state_[2].transposeInPlace();
			std::for_each(h_weight_.begin(), h_weight_.end(), [](Eigen::MatrixXf &wh) { wh.transposeInPlace(); });

			// begin concurrency for bptt
			start_ = limit_; end_ = start_ - slope_;

			// initialize thread pool
			for (int t = 0; t < thread_pool_.size(); t ++) {
				thread_pool_[t] = std::move(std::thread(&LSTM::time_step_update, this, start_, end_));
				start_ -= slope_; end_ -= slope_;
			}
			
			// join threads
			for (int t = 0; t < thread_pool_.size(); t ++) { 
				thread_pool_[t].join();
			}

			// perform the remaining time-steps
			if (output_2d_shape_[0] % thread_pool_.size() > 0) {
				time_step_update(start_, -1);
			}

			// calculate dJ/dX
			layer_2d_delta_ = dhfg_ * x_weight_[0].transpose() + dhig_ * x_weight_[1].transpose() + dhog_ * x_weight_[2].transpose() + dhmg_ * x_weight_[3].transpose();

			// perform change to parameters
			#pragma omp parallel 
			{
				#pragma omp for nowait 
				for (int j = 0; j < 4; j ++) {
					// bias momentum
					vb_[j] = 0.1 * vb_[j] + 0.9 * dbias_[j].array().square().sum();
					// X-weight momentum
					vw_[j] = 0.1 * vw_[j] + 0.9 * dx_weight_[j].array().square().sum();
					// H-weight momentum
					vh_[j] = 0.1 * vh_[j] + 0.9 * dh_weight_[j].array().square().sum();

					// change bias
					bias_[j] = bias_[j].array() - lrate_ / sqrtf(vb_[j] + erate_) * dbias_[j].array(); 

					// change x-weight
					x_weight_[j] = x_weight_[j].array() - lrate_ / sqrtf(vw_[j] + erate_) * dx_weight_[j].array();

					// change h-weight
					h_weight_[j] = h_weight_[j].array() - lrate_ / sqrtf(vh_[j] + erate_) * dh_weight_[j].array(); 
				}
			}	

			// re-transpose input, cp, and weight matrices in place
			input_.transposeInPlace();
			state_[2].transposeInPlace();
			std::for_each(h_weight_.begin(), h_weight_.end(), [](Eigen::MatrixXf &wh) { wh.transposeInPlace(); });

			// create async processes
			auto reset_dbias = std::async(std::launch::deferred, &LSTM::reset_mat, this, std::ref(dbias_));
			
			auto reset_dx_weight = std::async(std::launch::deferred, &LSTM::reset_mat, this, std::ref(dx_weight_));
			
			auto reset_dh_weight = std::async(std::launch::deferred, &LSTM::reset_mat, this, std::ref(dh_weight_));
			
			// reset gradient matrices asynchronously
			reset_dbias.get(); reset_dx_weight.get(); reset_dh_weight.get();
		}

		const std::string type() { return "2D"; }

		const std::string name() { return "LSTM"; }

		void reset() override {
			state_[0].row(limit_) = state_[0].row(limit_).array() * 0.0; 
			layer_2d_output_.row(limit_) = layer_2d_output_.row(limit_).array() * 0.0; 
		}

		void set_delta(const Eigen::MatrixXf &prev_layer_delta) override {
			// dH = dJ (l + 1) as LSTM does not have an appropriate general dH 
			doutput_ = prev_layer_delta; 
		}

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
			doutput_ = layer_2d_output_ - Y;
			
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

			return (1.0 - Y.array() * layer_2d_output_.array()).log().sum();
		}

		void save(std::ofstream &file) {
			assert(init_flag_);
			
			// convert all parameters into vectors
			std::vector <Eigen::VectorXf> vx_bias, vx_weight, vh_weight; vx_bias.resize(4); vx_weight.resize(4); vh_weight.resize(4);

			// convert matrices via a loop
			for (int j = 0; j < 4; j ++) { 
				// bias
				vx_bias[j] = Eigen::Map <Eigen::VectorXf> (bias_[j].data(), bias_[j].size());

				// X-weight 
				vx_weight[j] = Eigen::Map <Eigen::VectorXf> (x_weight_[j].data(), x_weight_[j].size());

				// H-weight
				vh_weight[j] = Eigen::Map <Eigen::VectorXf> (h_weight_[j].data(), h_weight_[j].size());
			}

			// write to file
			file << "\nLSTM\n";
			file << "Activation: " << activation_ << '\n';	
			file << "Recurrent_Activation: " << recurrent_activation_ << '\n';
			file << "Learning_rate: (" << lrate_ << ")\n";
            file << "Momentum_rate: (" << erate_ << ")\n";
			file << "Input_shape: [" << input_2d_shape_[0] << ',' << input_2d_shape_[1] << "]\n";
			file << "Output_shape: [" << output_2d_shape_[0] << ',' << output_2d_shape_[1] << "]\n";

			file << "Forget_gate_bias: [";
			write_vector(vx_bias[0], file);
			file << "Forget_gate_weight: [";
			write_vector(vx_weight[0], file);	
			file << "Forget_gate_uweight: [";
			write_vector(vh_weight[0], file);

			file << "Input_gate_bias: [";
			write_vector(vx_bias[1], file);
			file << "Input_gate_weight: [";
			write_vector(vx_weight[1], file);	
			file << "Input_gate_uweight: [";
			write_vector(vh_weight[1], file);

			file << "Output_gate_bias: [";
			write_vector(vx_bias[2], file);
			file << "Output_gate_weight: [";
			write_vector(vx_weight[2], file);	
			file << "Output_gate_uweight: [";
			write_vector(vh_weight[2], file);

			file << "Memory_gate_bias: [";
			write_vector(vx_bias[3], file);
			file << "Memory_gate_weight: [";
			write_vector(vx_weight[3], file);	
			file << "Memory_gate_uweight: [";
			write_vector(vh_weight[3], file);
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
			output_2d_shape_[0] = input_2d_shape_[0];
			
			// set the shape of layer delta, and layer output	
			layer_2d_delta_ = Eigen::MatrixXf::Zero(input_2d_shape_[0], input_2d_shape_[1]);
			layer_2d_output_ = Eigen::MatrixXf::Zero(input_2d_shape_[0], output_2d_shape_[1]);
				
			// set the shape of d(activated(C(t))		
			dh_cell_state_ = Eigen::MatrixXf::Zero(input_2d_shape_[0], output_2d_shape_[1]);
			
			// load biases
			bias_[0] = Eigen::Map<const Eigen::MatrixXf> (bias_fgate.data(), 1, output_2d_shape_[1]); 
			bias_[1] = Eigen::Map<const Eigen::MatrixXf> (bias_igate.data(), 1, output_2d_shape_[1]);
			bias_[2] = Eigen::Map<const Eigen::MatrixXf> (bias_ogate.data(), 1, output_2d_shape_[1]); 
			bias_[3] = Eigen::Map<const Eigen::MatrixXf> (bias_cgate.data(), 1, output_2d_shape_[1]);

			// load X-weights
			x_weight_[0] = Eigen::Map<const Eigen::MatrixXf> (weight_fgate.data(), input_2d_shape_[1], output_2d_shape_[1]);
			x_weight_[1] = Eigen::Map<const Eigen::MatrixXf> (weight_igate.data(), input_2d_shape_[1], output_2d_shape_[1]);
			x_weight_[2] = Eigen::Map<const Eigen::MatrixXf> (weight_ogate.data(), input_2d_shape_[1], output_2d_shape_[1]);
			x_weight_[3] = Eigen::Map<const Eigen::MatrixXf> (weight_cgate.data(), input_2d_shape_[1], output_2d_shape_[1]);

			// load H-weights 
			h_weight_[0] = Eigen::Map<const Eigen::MatrixXf> (u_weight_fgate.data(), output_2d_shape_[1], output_2d_shape_[1]);
			h_weight_[1] = Eigen::Map<const Eigen::MatrixXf> (u_weight_igate.data(), output_2d_shape_[1], output_2d_shape_[1]);
			h_weight_[2] = Eigen::Map<const Eigen::MatrixXf> (u_weight_ogate.data(), output_2d_shape_[1], output_2d_shape_[1]);
			h_weight_[3] = Eigen::Map<const Eigen::MatrixXf> (u_weight_cgate.data(), output_2d_shape_[1], output_2d_shape_[1]);
			
			// states, and outputs
			std::for_each(state_.begin(), state_.end(), [this](auto &mat) { mat = Eigen::MatrixXf::Zero(input_2d_shape_[0], output_2d_shape_[1]); });

			std::for_each(gate_.begin(), gate_.end(), [this](auto &mat) { mat = Eigen::MatrixXf::Zero(input_2d_shape_[0], output_2d_shape_[1]); });
			
			std::for_each(dgate_.begin(), dgate_.end(), [this](auto &mat) { mat = Eigen::MatrixXf::Zero(input_2d_shape_[0], output_2d_shape_[1]); });

			// set shape of gradients 
			std::for_each(dbias_.begin(), dbias_.end(), [this](auto &mat) { mat = Eigen::MatrixXf::Zero(1, output_2d_shape_[1]); });
			
			std::for_each(dx_weight_.begin(), dx_weight_.end(), [this](auto &mat) { mat = Eigen::MatrixXf::Zero(input_2d_shape_[1], output_2d_shape_[1]); }); 
			
			std::for_each(dh_weight_.begin(), dh_weight_.end(), [this](auto &mat) { mat = Eigen::MatrixXf::Zero(output_2d_shape_[1], output_2d_shape_[1]); });

			// select the appropriate size of thread pool and calculate the spread in thread executions	
			thread_pool_.resize(input_2d_shape_[0] <= std::thread::hardware_concurrency() ? input_2d_shape_[0] : input_2d_shape_[0] % std::thread::hardware_concurrency() + 1);
			
			slope_ = input_2d_shape_[0] / thread_pool_.size();
		}

	private:
		int limit_;																	// sequence length - 1

		std::string recurrent_activation_;											// names of activation function	

		std::vector <float> vw_ {4, 0.0}, vh_ {4, 0.0}, vb_ {4, 0.0};				// momentum values of each parameter in the layer

		std::vector <Eigen::MatrixXf> bias_;										// bias matrices of each gate

		std::vector <Eigen::MatrixXf> dbias_;										// gradients of each bias matrix

		std::vector <Eigen::MatrixXf> x_weight_;									// input weights of each gate

		std::vector <Eigen::MatrixXf> dx_weight_;									// gradients of each input weight

		std::vector <Eigen::MatrixXf> h_weight_;									// recurrent weights of each gate

		std::vector <Eigen::MatrixXf> dh_weight_;									// gradients of each recurrent weight

		std::vector <Eigen::MatrixXf> gate_;				 						// forget, input, output, and memory gates

		std::vector <Eigen::MatrixXf> dgate_;										// gradients of each gate

		std::vector <Eigen::MatrixXf> state_;										// c(t), activated c(t), c(t - 1), h(t - 1)

		Eigen::MatrixXf input_, doutput_, dh_cell_state_;							// X, dJ/dH, H(t - 1), C(t), activated C(t), C(t - 1)

		Eigen::MatrixXf dcfg_, dcig_, dcmg_, dhfg_, dhig_, dhog_, dhmg_;

        std::function <void (Eigen::MatrixXf &, Eigen::MatrixXf &, const int &)> cell_act_fun_, gate_act_fun_;	// empty functions for activation and recurrent activation functions

		void reset_mat(std::vector <Eigen::MatrixXf> &v) {
			// loop through vector and reset matrix
			std::for_each(v.begin(), v.end(), [](Eigen::MatrixXf &mat) { mat = mat.array() * 0.0; });
		}

		void time_step_update(int s, const int end) {
			// Create end time step, dHprev, dCprev
			int e;
			Eigen::MatrixXf dtfg, dtig, dtog, dtmg;		// gradient of each gate during bptt
			Eigen::MatrixXf dhp = Eigen::MatrixXf::Zero(1, output_2d_shape_[1]), dcp = Eigen::MatrixXf::Zero(1, output_2d_shape_[1]);	

			for (s; s > end; s --) {
				e = randint <int> (s / 2 - 1, s);	

				dbias_[0] =	dbias_[0] + dhfg_.row(s); 
				dbias_[1] = dbias_[1] + dhig_.row(s);
				dbias_[2] =	dbias_[2] + dhog_.row(s); 
				dbias_[3] = dbias_[3] + dhmg_.row(s);
			
				dx_weight_[0] = dx_weight_[0] + (input_.col(s) * dhfg_.row(s));
				dx_weight_[1] = dx_weight_[1] + (input_.col(s) * dhig_.row(s)); 
				dx_weight_[2] = dx_weight_[2] + (input_.col(s) * dhog_.row(s));
				dx_weight_[3] = dx_weight_[3] + (input_.col(s) * dhmg_.row(s));
				
				dh_weight_[0] = dh_weight_[0] + (state_[3].col(s) * dhfg_.row(s));
				dh_weight_[1] = dh_weight_[1] + (state_[3].col(s) * dhig_.row(s));
				dh_weight_[2] = dh_weight_[2] + (state_[3].col(s) * dhog_.row(s));
				dh_weight_[3] = dh_weight_[3] + (state_[3].col(s) * dhmg_.row(s));

				dhp = dhfg_.row(s) * h_weight_[0] + dhig_.row(s) * h_weight_[1] + dhog_.row(s) * h_weight_[2] + dhmg_.row(s) * h_weight_[3];

				dcp = doutput_.row(s).cwiseProduct(gate_[2].row(s)).cwiseProduct(dh_cell_state_.row(s)).cwiseProduct(gate_[0].row(s));

				for (int t = s - 1; t > e; t --) {
					// calculate gradient of each gate for each time step
					dtfg = dhp.cwiseProduct(dgate_[0].row(t)) + dcp.cwiseProduct(dcfg_.row(t));
					dtig = dhp.cwiseProduct(dgate_[1].row(t)) + dcp.cwiseProduct(dcig_.row(t));
					dtmg = dhp.cwiseProduct(dgate_[3].row(t)) + dcp.cwiseProduct(dcmg_.row(t));
					dtog = dhp.cwiseProduct(dgate_[2].row(t));

					dbias_[0] = dbias_[0] + dtfg;	
					dbias_[1] = dbias_[1] + dtig;
					dbias_[3] = dbias_[3] + dtmg;
					dbias_[2] = dbias_[2] + dtog; 

					dx_weight_[0] = dx_weight_[0] + (input_.col(t) * dtfg);
					dx_weight_[1] = dx_weight_[1] + (input_.col(t) * dtig);
					dx_weight_[3] = dx_weight_[3] + (input_.col(t) * dtmg);
					dx_weight_[2] = dx_weight_[2] + (input_.col(t) * dtog);

					dh_weight_[0] = dh_weight_[0] + (state_[3].col(t) * dtfg);
					dh_weight_[1] = dh_weight_[1] + (state_[3].col(t) * dtig);
					dh_weight_[3] = dh_weight_[3] + (state_[3].col(t) * dtmg);
					dh_weight_[2] = dh_weight_[2] + (state_[3].col(t) * dtog);
					
					if (t != e + 1) {	
						dhp = dtfg * h_weight_[0] + dtig * h_weight_[1] + dtog * h_weight_[2] + dtmg * h_weight_[3];

						dcp = dcp.cwiseProduct(gate_[0].row(t));
					}
				}	
			}
		}
};	

}

#endif
