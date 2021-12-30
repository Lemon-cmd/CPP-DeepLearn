#ifndef GRU_HPP
#define GRU_HPP 

#include "layer.hpp" 

namespace dpp {

class GRU : public Layer {
	
	/* Gated Recurrent Unit Layer */

	public:
		GRU (const int neurons = 1, const std::string activation = "tanh", const std::string recurrent_activation = "sigmoid", 
			const float lrate = 0.001, const float erate = 1e-8) {
			
			/* Save the parameters, and initalize parallelism and set the appropriate activation and recurrent activation functions */
			assert(neurons > 0 && lrate > 0.0f && erate > 0.0f);

			// set learning and optimization rate; save neurons to output shape
			lrate_ = lrate; erate_ = erate;
			out_dim_[1] = neurons;

			// set size of vectors
			state_.resize(3); gate_.resize(3); dgate_.resize(3); bias_.resize(3); x_weight_.resize(3); h_weight_.resize(3);

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

		const std::string name() { return "GRU"; }

		const MatrixXf& Get2dDelta() override { return delta_; } 

		const MatrixXf& Get2dOutput() override { return output_; }

		const Dim2d Get2dOutputDim() override { return out_dim_; } 

		void Reset() override {
			static const int limit = in_dim_[0] - 1;
			output_.row(limit) = output_.row(limit).array() * 0.0; 
		}

		void SetDelta(const MatrixXf &delta) override {
			dh_ = delta;																								// dH = dJ (l + 1) as LSTM does not have a general dH 
		}
	
		void Topology() {
			std::cout << "\n\t------------- GRU Layer ---------------\n"
					  << "\n\tParameters:\t(" << out_dim_[1] * out_dim_[1] * 3 + in_dim_[1] * out_dim_[0] * 3 + out_dim_[1] * 3 << ')'
					  << "\n\tInput-dim:\t(" << in_dim_[0] << ", " << in_dim_[1] << ')'
                      << "\n\tOutput-dim:\t(" << out_dim_[0] << ", " << out_dim_[1] << ")\n"
					  << "\n\t----------------------------------------\n\n";
		}

		void init(const Dim2d in_dim) override {
			assert(in_dim[0] > 0 && in_dim[1] > 0);
            /*
             * Initialize the parameters of the layer and save the input shape
             */

            // set initialized flag
            init_flag_ = true;
			
			// save input and output shapes
            in_dim_ = in_dim; out_dim_[0] = in_dim[0];

            // set shape of layer delta and layer output
			delta_ = MatrixXf::Zero(in_dim_[0], in_dim_[1]);
			output_ = MatrixXf::Zero(in_dim_[0], out_dim_[1]);
			
			// set the shape of H(t - 1)) 
			std::for_each(state_.begin(), state_.end(), [this](auto &mat) { mat = MatrixXf::Zero(in_dim_[0], out_dim_[1]); });

			// set the shape of each gate and their gradient 
			std::for_each(gate_.begin(), gate_.end(), [this](auto &mat) { mat = MatrixXf::Zero(in_dim_[0], out_dim_[1]); });
			std::for_each(dgate_.begin(), dgate_.end(), [this](auto &mat) { mat = MatrixXf::Zero(in_dim_[0], out_dim_[1]); });

			// initialize biases and their gradient
			std::for_each(bias_.begin(), bias_.end(), [this](auto &mat) { mat = MatrixXf::Ones(1, out_dim_[1]); });

			// initialize X-weight and their gradient 
			std::for_each(x_weight_.begin(), x_weight_.end(), [this](auto &mat) { mat = MatrixXf::Constant(in_dim_[1], out_dim_[1], 
																					   2.0f / sqrtf(in_dim_[1] * out_dim_[1])); });

			// initialize H-weight and their gradient
			std::for_each(h_weight_.begin(), h_weight_.end(), [this](auto &mat) { mat = MatrixXf::Constant(out_dim_[1], out_dim_[1], 
																					   2.0f / out_dim_[1]); });	

			// select the appropriate size of thread pool and calculate the spread in thread executions	
			thread_pool_.resize(in_dim_[0] <= std::thread::hardware_concurrency() ? in_dim_[0] : in_dim_[0] % std::thread::hardware_concurrency() + 1);
			slope_ = in_dim_[0] / thread_pool_.size();
		}
		
		void Forward(const MatrixXf &X) override {
			assert(X.rows() == in_dim_[0] && X.cols() == in_dim_[1] && init_flag_);

			input_ = X;																										// store X
			static const int limit = in_dim_[0] - 1;																		// sequence length - 1
			state_[2].row(0) = output_.row(limit);																			// store H(t - 1)

			// loop through sequence 
			for (int w = 0; w < in_dim_[0]; w ++) {
				// calculate Z of each gate 
				gate_[0].row(w) = input_.row(w) * x_weight_[0] + state_[2].row(w) * h_weight_[0] + bias_[0];				// update gate
				gate_[1].row(w) = input_.row(w) * x_weight_[1] + state_[2].row(w) * h_weight_[1] + bias_[1];				// reset gate 

				reactivate_(gate_[0], dgate_[0], w);																		// activate update gate  
				reactivate_(gate_[1], dgate_[1], w);																		// activate reset gate 

				state_[0].row(w) = gate_[1].row(w).cwiseProduct(state_[2].row(w));											// r(t) * h(t - 1)

				gate_[2].row(w) = input_.row(w) * x_weight_[2] + state_[0].row(w) * h_weight_[2] + bias_[2];				// candidate gate
				activate_(gate_[2], dgate_[2], w);																			// activate candidate gate

				state_[1].row(w) = 1.0 - gate_[0].row(w).array();															// 1.0 - z(t)

				// calculate h(t)
				output_.row(w) = state_[1].row(w).cwiseProduct(state_[2].row(w)) 
							   + gate_[0].row(w).cwiseProduct(gate_[2].row(w));

				if (w < limit) { state_[2].row(w + 1) = output_.row(w);	}													// set h (t - 1)
			}
		}

		void Update() {
			static MatrixXfVec dhg, db, dxw, dhw; 
			dhg.resize(3); db.resize(3); dxw.resize(3); dhw.resize(3);
		
			#pragma omp parallel 	
			{ 
				#pragma omp for nowait
				for (int j = 0; j < 3; j ++) {
					db[j] = MatrixXf::Zero(1, out_dim_[1]);
					dxw[j] = MatrixXf::Zero(in_dim_[1], out_dim_[1]);
					dhw[j] = MatrixXf::Zero(out_dim_[1], out_dim_[1]);
					h_weight_[j].transposeInPlace();																		// transpose all h-weight
				}
			}

			dgate_[2] = dgate_[2].cwiseProduct(gate_[0]);																	// calculate dh(t)/dc(t)
			dgate_[0] = dgate_[0].cwiseProduct(-state_[2] + gate_[2]);														// calculate dh(t)/dz(t)
			dgate_[1] = dgate_[1].cwiseProduct(dgate_[2]).cwiseProduct(state_[2] * h_weight_[2]);							// calculate dh(t)/dr(t)
			
			input_.transposeInPlace();																						// transpose X
			state_[2].transposeInPlace();																					// transpose h(t - 1) 

			#pragma omp parallel 	
			{ 
				#pragma omp for nowait
				for (int j = 0; j < 3; j ++) {
					dhg[j] = dh_.cwiseProduct(dgate_[j]);																	 // dJ / dz of gate
				}
			}

			start_ = in_dim_[0] - 1; end_ = start_ - slope_;																// thread index range
			for (int t = 0; t < thread_pool_.size(); t ++) {																// create thread pool
				thread_pool_[t] = std::move(std::thread(&GRU::time_step_update, this, 
														std::ref(dhg), std::ref(db), std::ref(dxw), std::ref(dhw),
														start_, end_));														// create thread process
				start_ -= slope_; end_ -= slope_;																			// change thread index range
			}

			for (int t = 0; t < thread_pool_.size(); t ++) {																
				thread_pool_[t].join();																						// join threads
			}

			if (out_dim_[0] % thread_pool_.size() > 0) {																	// perform the remainding operations
				time_step_update(dhg, db, dxw, dhw, start_, -1);
			}

			delta_ = dhg[0] * x_weight_[0].transpose() 
				   + dhg[1] * x_weight_[1].transpose() + dhg[2] * x_weight_[2].transpose();									// dJ / dX 

			// perform change to parameters
			#pragma omp parallel 
			{
				#pragma omp for simd nowait 
				for (int j = 0; j < 3; j ++) {
					vb_[j] = 0.1 * vb_[j] + 0.9 * db[j].array().square().sum();												// bias momentum
					vw_[j] = 0.1 * vw_[j] + 0.9 * dxw[j].array().square().sum();											// X-weight momentum
					vh_[j] = 0.1 * vh_[j] + 0.9 * dhw[j].array().square().sum();											// H-weight momentum

					bias_[j] = bias_[j].array() - lrate_ / sqrtf(vb_[j] + erate_) * db[j].array();							// change bias
					x_weight_[j] = x_weight_[j].array() - lrate_ / sqrtf(vw_[j] + erate_) * dxw[j].array();					// change x-weight
					h_weight_[j] = h_weight_[j].array() - lrate_ / sqrtf(vh_[j] + erate_) * dhw[j].array();					// change h-weight
				}
			}	

			input_.transposeInPlace();																						// transpose X
			state_[2].transposeInPlace();                                                                                   // transpose h(t - 1) 
			std::for_each(std::execution::par_unseq, h_weight_.begin(), h_weight_.end(), 
						 [](auto &mat) { mat.transposeInPlace(); });														// transpose all of h-weight
		}

		const float MeanSquaredError(const MatrixXf &Y, float &accuracy) override {
			dh_ = output_ - Y;																								// dH = H - Y

		   	accuracy = output_.unaryExpr<float(*)(float)>(&roundf).isApprox(Y) ? accuracy + 1 : accuracy;
			
			return 0.5 * dh_.array().square().sum();
		}
		
		const float CrossEntropyError(const MatrixXf &Y, float &accuracy) override {
			static float hits; hits = 0.0;																					// matches of each row
			
			if (activation_ != "softmax") { dh_ = -Y.array() / output_.array() * dh_.array(); }								// set new dJ
			else { dh_ = output_ - Y; }

			start_ = 0; end_ = slope_;																						// initalize start and end index of thread
			
			// measure accuracy of H concurrently
			for (int t = 0; t < thread_pool_.size(); t ++) {																	
				thread_pool_[t] = std::thread(&GRU::Measure, this, 
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
			std::vector <VectorXf> vx_bias, vx_weight, vh_weight; vx_bias.resize(3); vx_weight.resize(3); vh_weight.resize(3);

			// convert matrices into vectors via a loop
			for (int j = 0; j < 3; j ++) { 
				vx_bias[j] = Eigen::Map <VectorXf> (bias_[j].data(), bias_[j].size());
				vx_weight[j] = Eigen::Map <VectorXf> (x_weight_[j].data(), x_weight_[j].size());
				vh_weight[j] = Eigen::Map <VectorXf> (h_weight_[j].data(), h_weight_[j].size());
			}
			
			// write to file
			file << "\nGRU\n";
			file << "Activation: " << activation_ << '\n';	
			file << "Recurrent_Activation: " << recurrent_activation_ << '\n';
			file << "Learning_rate: (" << lrate_ << ")\n";
            file << "Momentum_rate: (" << erate_ << ")\n";
			file << "Input_shape: [" << in_dim_[0] << ',' << in_dim_[1] << "]\n";
			file << "Output_shape: [" << out_dim_[0] << ',' << out_dim_[1] << "]\n";
			
			file << "Update_gate_bias: [";	WriteVector(vx_bias[0], file);
			file << "Update_gate_weight: ["; WriteVector(vx_weight[0], file);
			file << "Update_gate_uweight: ["; WriteVector(vh_weight[0], file);
			
			file << "Reset_gate_bias: [";	WriteVector(vx_bias[1], file);
			file << "Reset_gate_weight: ["; WriteVector(vx_weight[1], file);
			file << "Reset_gate_uweight: ["; WriteVector(vh_weight[1], file);
			
			file << "Candidate_gate_bias: [";	WriteVector(vx_bias[2], file);
			file << "Candidate_gate_weight: ["; WriteVector(vx_weight[2], file);
			file << "Candidate_gate_uweight: ["; WriteVector(vh_weight[2], file);
		}
		
		void Load(const Dim2d &in_dim, const Dim2d &out_dim, const std::vector<VectorXf> &weight) override {
			assert(weight.size() == 9);	
		
			// set the parameters to metadata inside of the layer
			init_flag_ = true;			
			
			// save input and output shapes
			in_dim_ = in_dim; out_dim_ = out_dim;
			
			// set the shape of layer delta, and layer output	
			delta_ = MatrixXf::Zero(in_dim_[0], in_dim_[1]);
			output_ = MatrixXf::Zero(in_dim_[0], out_dim_[1]);
				
			// load biases
			for (int j = 0; j < 3; j ++) { 
				bias_[j] = Eigen::Map <const MatrixXf> (weight[j].data(), 1, out_dim_[1]);
			}

			// load X-weight
			for (int j = 3; j < 6; j ++) { 
				x_weight_[j] = Eigen::Map <const MatrixXf> (weight[j].data(), in_dim_[1], out_dim_[1]);
			}
		
			// load H-weight 
			for (int j = 6; j < 9; j ++) {
				h_weight_[j] = Eigen::Map <const MatrixXf> (weight[j].data(), out_dim_[1], out_dim_[1]);
			}
			
			// states, and outputs
			std::for_each(gate_.begin(), gate_.end(), [this](auto &mat) { mat = MatrixXf::Zero(in_dim_[0], out_dim_[1]); });
			
			std::for_each(dgate_.begin(), dgate_.end(), [this](auto &mat) { mat = MatrixXf::Zero(in_dim_[0], out_dim_[1]); });
			
			std::for_each(state_.begin(), state_.end(), [this](auto &mat) { mat = MatrixXf::Zero(in_dim_[0], out_dim_[1]); });

			// select the appropriate size of thread pool and calculate the spread in thread executions	
			thread_pool_.resize(in_dim_[0] <= std::thread::hardware_concurrency() ? in_dim_[0] : in_dim_[0] % std::thread::hardware_concurrency() + 1);
			slope_ = in_dim_[0] / thread_pool_.size();
		}

	private:
		Dim2d in_dim_, out_dim_;													// input and output dimensions
		
		std::string recurrent_activation_;											// names of activation function	
		
		MatrixXfVec state_;															// r(t) * h(t - 1), 1 - z(t), h(t - 1)

		MatrixXfVec gate_, dgate_;													// forget, input, output, and memory gates and their gradient
	
		MatrixXfVec bias_, x_weight_, h_weight_;									// weights of layer

		MatrixXf input_, output_, delta_, dh_;										// input, output, layer gradient, dJ/dH						

		std::vector <float> vw_ {3, 1.0}, vh_ {3, 1.0}, vb_ {3, 1.0};				// momentum values of each parameter in the layer

		std::function <void (MatrixXf &, MatrixXf &, const int &)> activate_, reactivate_;	// empty functions for activation and recurrent activation functions

		void time_step_update(MatrixXfVec &dhg, MatrixXfVec &db, 
							  MatrixXfVec &dxw, MatrixXfVec &dhw, int s, const int end) {
			int e;
			MatrixXf dhp;
			MatrixXfVec dzp; dzp.resize(3);	

			for (s; s > end; s --) {
				e = randint <int> (-1, s);
				
				#pragma omp parallel 
				{
					#pragma omp for simd nowait 
					for (int j = 0; j < 3; j ++) {
						db[j] = db[j] + dhg[j].row(s);																		// gradient of gate-bias
						dxw[j] = dxw[j] + input_.col(s) * dhg[j].row(s);													// gradient of gate x-weight
						dhw[j] = dhw[j] + state_[2].col(s) * dhg[j].row(s);													// gradient of gate h-weight
					}
				}	
				
				dhp = dh_.row(s).cwiseProduct(state_[0].row(s))																// dh(t) / dh(t-1)
					+ dhg[0].row(s) * h_weight_[0] + dhg[1].row(s) * h_weight_[1] + dhg[2].row(s) * h_weight_[2]; 

				for (int t = s - 1; t > e; t --) {
					#pragma omp parallel 
					{
						#pragma omp for simd nowait 
						for (int j = 0; j < 3; j ++) {
							dzp[j] = dhp.cwiseProduct(dgate_[j].row(t));													// gradient previous time-step gate
							db[j] = db[j] + dzp[j];																			// gradient of p-time step bias
							dxw[j] = dxw[j] + input_.col(t) * dzp[j];														// gradient of p-time step x-weight
							dhw[j] = dhw[j] + state_[2].col(t) * dzp[j];													// gradient of p-time step h-weight
						}
					}	

					if (t != e + 1) {
						dhp = dhp.cwiseProduct(state_[0].row(t))															 
							+ dzp[0] * h_weight_[0] + dzp[1] * h_weight_[1] + dzp[2] * h_weight_[2];						// set new dh(t) / dh(t - 1)
					}				
				}
			}
		}
};

}

#endif 
