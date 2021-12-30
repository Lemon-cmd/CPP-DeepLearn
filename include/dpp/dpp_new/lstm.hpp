#ifndef LSTM_HPP
#define LSTM_HPP

#include "layer.hpp" 

namespace dpp {

class LSTM : public Layer {
	/* Long Short Term Memory Layer */
	public:
		LSTM(const int neurons = 1, const std::string activation = "tanh", const std::string recurrent_activation = "sigmoid", 
			const float lrate = 0.001, const float erate = 1e-8) {

			/* Save the parameters, and initalize parallelism and set the appropriate activation and recurrent activation functions */
			assert(neurons > 0 && lrate > 0.0f && erate > 0.0f);		
			
			// etc parameters
			lrate_ = lrate; erate_ = erate;
			out_dim_[1] = neurons;
			
			// set size of layer's parameter-vectors	
			state_.resize(4); gate_.resize(4); dgate_.resize(4); x_weight_.resize(4); h_weight_.resize(4); bias_.resize(4); 

			// save the activation name for when saving to a file
			activation_ = activation == "normal" ? activation : activation == "softsign" ? activation : activation == "softmax" ? activation : activation == "relu" ? \
						  activation : activation == "sigmoid" ? activation : "tanh";

			// save the recurrent activation name for when saving to a file
			recurrent_activation_ = recurrent_activation == "normal" ? recurrent_activation : recurrent_activation == "tanh" ? recurrent_activation : recurrent_activation == "softmax" ? \
								    recurrent_activation : recurrent_activation == "relu" ? recurrent_activation : recurrent_activation == "softsign" ? recurrent_activation : "sigmoid";
			
			// make any necessary changes to the activation functions
			SetActivation(activate_, activation_);
			SetActivation(reactivate_, recurrent_activation_);
		}

		const std::string type() { return "2D"; }

		const std::string name() { return "LSTM"; }
		
		const MatrixXf& Get2dDelta() override { return delta_; } 

		const MatrixXf& Get2dOutput() override { return output_; }

		const Dim2d Get2dOutputDim() override { return out_dim_; } 
	
		void Reset() override {
			static const int limit = in_dim_[0] - 1;
			output_.row(limit) = output_.row(limit).setZero(); 
			state_[0].row(limit) = state_[0].row(limit).setZero(); 
		}
	
		void SetDelta(const MatrixXf &delta) override {
			dh_ = delta;
		}

		void Topology() {
			std::cout << "\n\t------------- LSTM Layer ---------------\n"
					  << "\n\tParameters:\t(" << out_dim_[1] * out_dim_[1] * 4 + in_dim_[1] * out_dim_[0] * 4 + out_dim_[1] * 4 << ')'
					  << "\n\tInput-dim:\t(" << in_dim_[0] << ", " << in_dim_[1] << ')'
                      << "\n\tOutput-dim:\t(" << out_dim_[0] << ", " << out_dim_[1] << ")\n"
					  << "\n\t----------------------------------------\n\n";
		}
		
		void init(const Dim2d in_dim) override {
		/* Initialize the parameters of the layer and save the input shape */
			assert(in_dim[0] > 0 && in_dim[1] > 0);
			// set initialized flag
			init_flag_ = true;

			// save shapes
			in_dim_ = in_dim; out_dim_[0] = in_dim_[0];

			// set shape of layer delta and layer output
			delta_ = MatrixXf::Zero(in_dim_[0], in_dim_[1]);
			output_ = MatrixXf::Zero(in_dim_[0], out_dim_[1]);

			// set the shape of d(activated(C(t)) and other states (C(t), activated C(t), C(t - 1), H(t - 1))
			dh_ct_ = MatrixXf::Zero(in_dim_[0], out_dim_[1]);
			std::for_each(state_.begin(), state_.end(), [this](auto &mat) { mat = MatrixXf::Zero(in_dim_[0], out_dim_[1]); });

			// set the shape of each gate and their gradient
			std::for_each(gate_.begin(), gate_.end(), [this](auto &mat) { mat = MatrixXf::Zero(in_dim_[0], out_dim_[1]); });
			std::for_each(dgate_.begin(), dgate_.end(), [this](auto &mat) { mat = MatrixXf::Zero(in_dim_[0], out_dim_[1]); });

			// initialize weight
			std::for_each(bias_.begin(), bias_.end(), [this](auto &mat) { mat = MatrixXf::Ones(1, out_dim_[1]); });
			std::for_each(h_weight_.begin(), h_weight_.end(), [this](auto &mat) { mat = MatrixXf::Constant(out_dim_[1], out_dim_[1], 2.0f / out_dim_[1]); });
			std::for_each(x_weight_.begin(), x_weight_.end(), [this](auto &mat) { mat = MatrixXf::Constant(in_dim_[1], out_dim_[1], 2.0f / sqrtf(in_dim_[1] * out_dim_[1])); });

			// select the appropriate size of thread pool and calculate the spread in thread executions
			thread_pool_.resize(in_dim_[0] <= std::thread::hardware_concurrency() ? in_dim_[0] : in_dim_[0] % std::thread::hardware_concurrency() + 1);
			slope_ = in_dim_[0] / thread_pool_.size();
		}

		void Forward(const MatrixXf &X) override {
			assert(X.rows() == in_dim_[0] && X.cols() == in_dim_[1] && init_flag_);
		
			input_ = X;																								// store X
			static const int limit = in_dim_[0] - 1;																// last index of sequence
			state_[3].row(0) = output_.row(limit);																	// store H(t - 1)
			state_[2].row(0) = state_[0].row(limit);																// store C(t - 1)

			for (int w = 0; w < in_dim_[0]; w ++) {
				// calculate Z of each gate
				gate_[0].row(w) = input_.row(w) * x_weight_[0] + state_[3].row(w) * h_weight_[0] + bias_[0];		// forget gate
				gate_[1].row(w) = input_.row(w) * x_weight_[1] + state_[3].row(w) * h_weight_[1] + bias_[1];		// input gate
				gate_[2].row(w) = input_.row(w) * x_weight_[2] + state_[3].row(w) * h_weight_[2] + bias_[2];		// output gate
				gate_[3].row(w) = input_.row(w) * x_weight_[3] + state_[3].row(w) * h_weight_[3] + bias_[3];		// memory gate
				
				// activate LSTM gates
				reactivate_(gate_[0], dgate_[0], w);																// forget gate 
				reactivate_(gate_[1], dgate_[1], w);																// input gate 
				reactivate_(gate_[2], dgate_[2], w);																// output gate 
				activate_(gate_[3], dgate_[3], w);																	// mem. gate 

				// calculate cell state : ft * c(t - 1) + it * gt  
				state_[1].row(w) = gate_[0].row(w).array() * state_[2].row(w).array() 
								 + gate_[1].row(w).array() * gate_[3].row(w).array();

				state_[0].row(w) = state_[1].row(w);																// save c(t)
				activate_(state_[1], dh_ct_, w);																	// activate c(t)
				output_.row(w) = gate_[2].row(w).cwiseProduct(state_[1].row(w));									// calculate h(t)
				
				if (w < limit) {
					state_[3].row(w + 1) = output_.row(w);															// set h(t - 1) for next t
					state_[2].row(w + 1) = state_[0].row(w);														// set c(t - 1) for next t
				}
			}
		}

		void Update() {
			static MatrixXfVec dc, db, dhg, dxw, dhw;
			dc.resize(3); db.resize(4); dhg.resize(4); dxw.resize(4); dhw.resize(4);
			
			#pragma omp parallel	
			{ 
				#pragma omp for simd nowait
				for (int j = 0; j < 4; j ++) {
					db[j] = MatrixXf::Zero(1, out_dim_[1]);
					dhg[j] = MatrixXf::Zero(in_dim_[0], out_dim_[1]);
					dxw[j] = MatrixXf::Zero(in_dim_[1], out_dim_[1]);
					dhw[j] = MatrixXf::Zero(out_dim_[1], out_dim_[1]);
					h_weight_[j].transposeInPlace();																		// transpose all h-weight
				}
			}

			dc[1] = dgate_[1].cwiseProduct(gate_[3]);																		// di(t)
			dc[2] = dgate_[3].cwiseProduct(gate_[1]);																		// dm(t)
			dc[0] = dgate_[0].cwiseProduct(state_[2]);																		// df(t)
			dgate_[2] = dgate_[2].cwiseProduct(state_[1]);																	// do(t)

			dgate_[0] = gate_[2].cwiseProduct(dh_ct_).cwiseProduct(dc[0]);													// dc(t) / df(t)
			dgate_[1] = gate_[2].cwiseProduct(dh_ct_).cwiseProduct(dc[1]);													// dc(t) / di(t)
			dgate_[3] = gate_[2].cwiseProduct(dh_ct_).cwiseProduct(dc[2]);													// dc(t) / dm(t)
	
			input_.transposeInPlace();																						// X.T
			state_[3].transposeInPlace();																					// H(t - 1).T

			delta_ = delta_.setZero();
			#pragma omp parallel 
			{
				#pragma omp for simd nowait 
				for (int j = 0; j < 4; j ++) { 
					dhg[j] = dh_.cwiseProduct(dgate_[j]);																	// dJ / gradient of gate respectively
					delta_ = delta_ + dhg[j] * x_weight_[j].transpose();													// calculate dJ/dX
				}
			}

			start_ = in_dim_[0] - 1; end_ = start_ - slope_;																// indices for thread
			for (int t = 0; t < thread_pool_.size(); t ++) {																// initialize thread pool
				thread_pool_[t] = std::move(std::thread(&LSTM::RUpdate, this, 
														std::ref(dc), std::ref(dhg), 
														std::ref(db), std::ref(dxw), std::ref(dhw),
														start_, end_));
				start_ -= slope_; end_ -= slope_;
			}

			for (int t = 0; t < thread_pool_.size(); t ++) {
				thread_pool_[t].join();																						// join threads
			}

			if (out_dim_[0] % thread_pool_.size() > 0) {
				RUpdate(dc, dhg, db, dxw, dhw, start_, -1);														// perform the remaining time-steps
			}

			// perform change to parameters
			#pragma omp parallel
			{
				#pragma omp for simd nowait
				for (int j = 0; j < 4; j ++) {
					vb_[j] = 0.1 * vb_[j] + 0.9 * db[j].array().square().sum();												// bias momentum
					vw_[j] = 0.1 * vw_[j] + 0.9 * dxw[j].array().square().sum();											// X-weight momentum
					vh_[j] = 0.1 * vh_[j] + 0.9 * dhw[j].array().square().sum();											// H-weight momentum

					bias_[j] = bias_[j].array() - lrate_ / sqrtf(vb_[j] + erate_) * db[j].array();							// change bias
					x_weight_[j] = x_weight_[j].array() - lrate_ / sqrtf(vw_[j] + erate_) * dxw[j].array();					// change x-weight
					h_weight_[j] = h_weight_[j].array() - lrate_ / sqrtf(vh_[j] + erate_) * dhw[j].array();					// change h-weight
				}
			}

			input_.transposeInPlace();																				
			state_[3].transposeInPlace();																				
			std::for_each(std::execution::par_unseq, h_weight_.begin(), h_weight_.end(), [](MatrixXf &wh) { wh.transposeInPlace(); });	
		}

		const float MeanSquaredError(const MatrixXf &Y, float &accuracy) override {
			dh_ = output_ - Y;																							// dH = H - Y

		   	accuracy = output_.unaryExpr<float(*)(float)>(&roundf).isApprox(Y) ? accuracy + 1 : accuracy;
			
			return 0.5 * dh_.array().square().sum();
		}

		const float CrossEntropyError(const MatrixXf &Y, float &accuracy) override {
			static float hits; hits = 0.0;																				// matches of each row
			
			dh_ = -Y.array() / output_.array();																			// set new dJ

			start_ = 0; end_ = slope_;																					// initalize start and end index of thread
			// measure accuracy of H concurrently
			for (int t = 0; t < thread_pool_.size(); t ++) {																	
				thread_pool_[t] = std::thread(&LSTM::Measure, this, 
								  std::ref(output_), std::ref(Y), std::ref(hits), start_, end_);
				start_ += slope_; end_ += slope_;
			}

			for (int t = 0; t < thread_pool_.size(); t ++) {
				thread_pool_[t].join();
			}

			if (out_dim_[0] % thread_pool_.size() > 0) {
				Measure(output_, Y, hits, start_, out_dim_[0]);															// perform calculation on the remainder if exists
			}

			accuracy = hits / out_dim_[0] + accuracy;																	// accuracy = average of matches argmax of H to Y
			return (1.0 - Y.array() * output_.array().log()).sum();
		}

		void Save(std::ofstream &file) {
			assert(init_flag_);
			
			// convert all parameters into vectors
			std::vector <VectorXf> vx_bias, vx_weight, vh_weight; vx_bias.resize(4); vx_weight.resize(4); vh_weight.resize(4);

			// convert matrices to vectors
			for (int j = 0; j < 4; j ++) { 
				vx_bias[j] = Eigen::Map <VectorXf> (bias_[j].data(), bias_[j].size());
				vx_weight[j] = Eigen::Map <VectorXf> (x_weight_[j].data(), x_weight_[j].size());
				vh_weight[j] = Eigen::Map <VectorXf> (h_weight_[j].data(), h_weight_[j].size());
			}

			// write to file
			file << "\nLSTM\n";
			file << "Activation: " << activation_ << '\n';	
			file << "Recurrent_Activation: " << recurrent_activation_ << '\n';
			file << "Learning_rate: (" << lrate_ << ")\n";
            file << "Momentum_rate: (" << erate_ << ")\n";
			file << "Input_shape: [" << in_dim_[0] << ',' << in_dim_[1] << "]\n";
			file << "Output_shape: [" << out_dim_[0] << ',' << out_dim_[1] << "]\n";

			file << "Forget_gate_bias: ["; WriteVector(vx_bias[0], file);
			file << "Forget_gate_weight: ["; WriteVector(vx_weight[0], file);	
			file << "Forget_gate_uweight: ["; WriteVector(vh_weight[0], file);

			file << "Input_gate_bias: ["; WriteVector(vx_bias[1], file);
			file << "Input_gate_weight: ["; WriteVector(vx_weight[1], file);	
			file << "Input_gate_uweight: ["; WriteVector(vh_weight[1], file);

			file << "Output_gate_bias: ["; WriteVector(vx_bias[2], file);
			file << "Output_gate_weight: ["; WriteVector(vx_weight[2], file);	
			file << "Output_gate_uweight: ["; WriteVector(vh_weight[2], file);

			file << "Memory_gate_bias: ["; WriteVector(vx_bias[3], file);
			file << "Memory_gate_weight: ["; WriteVector(vx_weight[3], file);	
			file << "Memory_gate_uweight: ["; WriteVector(vh_weight[3], file);
		}

		void Load(const Dim2d &in_dim, const Dim2d &out_dim, const std::vector <VectorXf> &weight) override {
			assert(weight.size() == 12);
			// set initialized flag
			init_flag_ = true;

			// save shapes
			in_dim_ = in_dim; out_dim_[0] = in_dim_[0];

			// set shape of layer delta and layer output
			delta_ = MatrixXf::Zero(in_dim_[0], in_dim_[1]);
			output_ = MatrixXf::Zero(in_dim_[0], out_dim_[1]);

			// set the shape of d(activated(C(t)) and other states (C(t), activated C(t), C(t - 1), H(t - 1))
			dh_ct_ = MatrixXf::Zero(in_dim_[0], out_dim_[1]);
			std::for_each(state_.begin(), state_.end(), [this](auto &mat) { mat = MatrixXf::Zero(in_dim_[0], out_dim_[1]); });

			// set the shape of each gate and their gradient
			std::for_each(gate_.begin(), gate_.end(), [this](auto &mat) { mat = MatrixXf::Zero(in_dim_[0], out_dim_[1]); });
			std::for_each(dgate_.begin(), dgate_.end(), [this](auto &mat) { mat = MatrixXf::Zero(in_dim_[0], out_dim_[1]); });

			// reload biases
			for (int j = 0; j < 4; j ++) {
				bias_[j] = Eigen::Map <const MatrixXf> (weight[j].data(), 1, out_dim_[1]);
			}

			// load X-weight
			for (int j = 4; j < 8; j ++) {
				x_weight_[j] = Eigen::Map <const MatrixXf> (weight[j].data(), in_dim_[1], out_dim_[1]);
			}

			// load H-weight 
			for (int j = 8; j < 12; j ++) {
				h_weight_[j] = Eigen::Map <const MatrixXf> (weight[j].data(), out_dim_[1], out_dim_[1]);
			}

			// select the appropriate size of thread pool and calculate the spread in thread executions
			thread_pool_.resize(in_dim_[0] <= std::thread::hardware_concurrency() ? in_dim_[0] : in_dim_[0] % std::thread::hardware_concurrency() + 1);
			slope_ = in_dim_[0] / thread_pool_.size();
		}

	private:
		Dim2d in_dim_, out_dim_;																		// input and output shapes 

		std::string recurrent_activation_;																// recurrent act. name

		MatrixXf input_, output_, delta_, dh_, dh_ct_;													// input and dH matrices

		std::vector<float> vw_ {4, 1.0}, vh_ {4, 1.0}, vb_ {4, 1.0};									// weight- and bias-momentum vectors
		
		MatrixXfVec x_weight_, h_weight_, bias_, gate_, dgate_, state_;									// weight, bias, gate ouutputs, state = [c(t), g(c(t)), c(t - 1), h(t - 1)] 

		std::function <void (MatrixXf &, MatrixXf &, const int &)> activate_, reactivate_;				// empty functions for activation and recurrent activation functions
	
		void RUpdate(MatrixXfVec &dc, MatrixXfVec &dhg, 
					 MatrixXfVec &db, MatrixXfVec &dxw, 
					 MatrixXfVec &dhw, int s, const int end) {						

			int e;																						// end time
			MatrixXfVec dzp; dzp.resize(4);																// gradient of each gate during bptt
			MatrixXf dhp = MatrixXf::Zero(1, out_dim_[1]),												// dH(t - 1) 
					 dcp = MatrixXf::Zero(1, out_dim_[1]);												// dC(t - 1)

			for (s; s > end; s --) {
				e = randint <int> (-1, s);	
				
				#pragma omp parallel 
				{
					#pragma omp for simd nowait
					for (int j = 0; j < 4; j ++) {
						db[j] = db[j] + dhg[j].row(s); 
						dxw[j] = dxw[j] + (input_.col(s) * dhg[j].row(s));
						dhw[j] = dhw[j] + (state_[3].col(s) * dhg[j].row(s));
					}
				}

				dhp = dhg[0].row(s) * h_weight_[0] + dhg[1].row(s) * h_weight_[1] + dhg[2].row(s) * h_weight_[2] + dhg[3].row(s) * h_weight_[3];
				dcp = dh_.row(s).cwiseProduct(gate_[2].row(s)).cwiseProduct(dh_ct_.row(s)).cwiseProduct(gate_[0].row(s));

				for (int t = s - 1; t > e; t --) {
					// calculate gradient of each gate for each time step
					dzp[0] = dhp.cwiseProduct(dgate_[0].row(t)) + dcp.cwiseProduct(dc[0].row(t));
					dzp[1] = dhp.cwiseProduct(dgate_[1].row(t)) + dcp.cwiseProduct(dc[1].row(t));
					dzp[3] = dhp.cwiseProduct(dgate_[3].row(t)) + dcp.cwiseProduct(dc[2].row(t));
					dzp[2] = dhp.cwiseProduct(dgate_[2].row(t));

					#pragma omp parallel 
					{
						#pragma omp for simd nowait 
						for (int j = 0; j < 4; j ++) {
							db[j] = db[j] + dzp[0];
							dxw[j] = dxw[j] + input_.col(t) * dzp[j];
							dhw[j] = dhw[j] + state_[3].col(t) * dzp[j];
						}
					}
					
					if (t > e + 1) {	
						dhp = dzp[0] * h_weight_[0] + dzp[1] * h_weight_[1] + dzp[2] * h_weight_[2] + dzp[3] * h_weight_[3];
						dcp = dcp.cwiseProduct(gate_[0].row(t));
					}
				}	
			}
		}
};
}
#endif
