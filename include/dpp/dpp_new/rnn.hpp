#ifndef RNN_HPP
#define RNN_HPP

#include "layer.hpp" 

namespace dpp {

class RNN : public Layer {
	/* Simple Recurrent Layer based on Elman's network */
	public:
		RNN(const int neurons = 1, const std::string activation = "tanh", const std::string recurrent_activation = "sigmoid",
			const float lrate = 0.001, const float erate = 1e-8) {
		/* Save the parameters, and initalize parallelism and set the appropriate activation and recurrent activation functions */
            assert (neurons > 0 && lrate > 0.0 && erate > 0.0);
			// etc parameters
			lrate_ = lrate;
			erate_ = erate;
			out_dim_[1] = neurons;
			
			// set size of layer's parameter-vectors	
			state_.resize(1); gate_.resize(1); dgate_.resize(1); x_weight_.resize(1); h_weight_.resize(2); bias_.resize(2); 
		
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

		const std::string name() { return "RNN"; }
		
		const MatrixXf& Get2dDelta() override { return delta_; } 

		const MatrixXf& Get2dOutput() override { return output_; }

		const Dim2d Get2dOutputDim() override { return out_dim_; } 
		
		void SetDelta(const MatrixXf &delta) override {
			dh_ = dh_.cwiseProduct(delta); 
        }
		
		void Reset() override {
			static const int limit = in_dim_[0] - 1;
			output_.row(limit) = output_.row(limit).setZero(); 
        }

		void Topology() {
			std::cout << "\n\t------------- RNN Layer ---------------\n"
					  << "\n\tParameters:\t(" << out_dim_[1] * out_dim_[1] * 2 + in_dim_[1] * out_dim_[0] * 1 + out_dim_[1] * 2 << ')'
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
			dh_ = MatrixXf::Zero(in_dim_[0], out_dim_[1]);
			delta_ = MatrixXf::Zero(in_dim_[0], in_dim_[1]);
			output_ = MatrixXf::Zero(in_dim_[0], out_dim_[1]);
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
			state_[0].row(0) = output_.row(limit);																	// store H(t - 1)
			
			for (int w = 0; w < in_dim_[0]; w ++) {
				gate_[0].row(w) = input_.row(w) * x_weight_[0] 
								+ state_[0].row(w) * h_weight_[0] + bias_[0];										// ht = Wx * X + Uh * y (t - 1) + Bh

				reactivate_(gate_[0], dgate_[0], w);																// activate ht
				
				output_.row(w) = gate_[0].row(w) * h_weight_[1] + bias_[1];											// yt = Wh * ht + By 

				activate_(output_, dh_, w);																			// activate yt

				if (w < limit) { state_[0].row(w + 1) = output_.row(w); }											// set the next y (t - 1)
			}

		}

		void Update() {
			static MatrixXfVec db, dxw, dhw; 
			db.resize(2); dxw.resize(1); dhw.resize(2);

			std::fill(std::execution::par_unseq, db.begin(), db.end(), MatrixXf::Zero(1, out_dim_[1]));
			std::fill(std::execution::par_unseq, dxw.begin(), dxw.end(), MatrixXf::Zero(in_dim_[1], out_dim_[1]));
			std::fill(std::execution::par_unseq, dhw.begin(), dhw.end(), MatrixXf::Zero(out_dim_[1], out_dim_[1]));

			input_.transposeInPlace(); gate_[0].transposeInPlace(); 
			state_[0].transposeInPlace(); h_weight_[0].transposeInPlace();
  
            delta_ = (dh_ * h_weight_[1].transpose()).cwiseProduct(dgate_[0]) * x_weight_[0].transpose();					// calculate layer's delta
			
			start_ = in_dim_[0] - 1; end_ = start_ - slope_;																// indices for thread
			for (int t = 0; t < thread_pool_.size(); t ++) {																// initialize thread pool
				thread_pool_[t] = std::move(std::thread(&RNN::time_step_update, this, 
														std::ref(db), std::ref(dxw), std::ref(dhw),
														start_, end_));
				start_ -= slope_; end_ -= slope_;
			}

			for (int t = 0; t < thread_pool_.size(); t ++) {
				thread_pool_[t].join();																						// join threads
			}

			if (out_dim_[0] % thread_pool_.size() > 0) {
				time_step_update(db, dxw, dhw, start_, -1);																	// perform the remaining time-steps
			}

			dxw[0] = dxw[0].array() / out_dim_[0];
            vw_[0] = 0.1 * vw_[0] + 0.9 * dxw[0].array().square().sum();
            x_weight_[0] = x_weight_[0].array() - lrate_ / sqrtf(vw_[0] + erate_) * dxw[0].array();
			
			#pragma omp parallel 
			{	
				#pragma omp for nowait
				for (int j = 0; j < 2; j ++) {
					db[j] = db[j].array() / out_dim_[0];
					dhw[j] = dhw[j].array() / out_dim_[0];
					vb_[j] = 0.1 * vb_[j] + 0.9 * db[j].array().square().sum();
					vh_[j] = 0.1 * vh_[j] + 0.9 * dhw[j].array().square().sum();
					bias_[j] = bias_[j].array() - lrate_ / sqrtf(vb_[j] + erate_) * db[j].array();
					h_weight_[j] = h_weight_[j].array() - lrate_ / sqrtf(vh_[j] + erate_) * dhw[j].array();      
				}
			}
			
		  	input_.transposeInPlace(); gate_[0].transposeInPlace();
            state_[0].transposeInPlace(); h_weight_[0].transposeInPlace();
		}

		const float MeanSquaredError(const MatrixXf &Y, float &accuracy) override {
			dh_ = (output_ - Y).cwiseProduct(dh_);																		// dH = H - Y
		   	accuracy = output_.unaryExpr<float(*)(float)>(&roundf).isApprox(Y) ? accuracy + 1 : accuracy;
			return 0.5 * dh_.array().square().sum();
		}

		const float CrossEntropyError(const MatrixXf &Y, float &accuracy) override {
			static float hits; hits = 0.0;																				// matches of each row

			if (activation_ != "softmax") { dh_ = -Y.array() / output_.array() * dh_.array(); }							// set new dJ
			else { dh_ = output_ - Y; }

			start_ = 0; end_ = slope_;																					// initalize start and end index of thread
			for (int t = 0; t < thread_pool_.size(); t ++) {															// measure accuracy of H concurrently
				thread_pool_[t] = std::thread(&RNN::Measure, this, 
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
            Eigen::Map <VectorXf> bv_igate (bias_[0].data(), bias_[0].size()),
										 wv_igate (x_weight_[0].data(), x_weight_[0].size()),
                                         uv_igate (h_weight_[0].data(), h_weight_[0].size());
  
            Eigen::Map <VectorXf> bv_out (bias_[1].data(), bias_[1].size()),
                                         wv_out (h_weight_[1].data(), h_weight_[1].size());
            file << "\nRNN\n";
            file << "Activation: " << activation_ << '\n';
            file << "Recurrent_Activation: " << recurrent_activation_ << '\n';
            file << "Learning_rate: (" << lrate_ << ")\n";
            file << "Momentum_rate: (" << erate_ << ")\n";
            file << "Input_shape: [" << in_dim_[0] << ',' << in_dim_[1] << "]\n";
            file << "Output_shape: [" << out_dim_[0] << ',' << out_dim_[1] << "]\n";
            file << "Input_gate_bias: [";
            WriteVector(bv_igate, file);
            file << "Input_gate_weight: [";
            WriteVector(wv_igate, file);
            file << "Input_gate_uweight: [";
            WriteVector(uv_igate, file);
            file << "Output_bias: [";
            WriteVector(bv_out, file);
            file << "Output_weight: [";
            WriteVector(wv_out, file);
		}

		void Load(const Dim2d &in_dim, const Dim2d &out_dim, const std::vector <VectorXf> &weight) override {
			assert(weight.size() == 5);

			init_flag_ = true;
			in_dim_ = in_dim; out_dim_[0] = out_dim[0];

			dh_ = MatrixXf::Zero(in_dim_[0], out_dim_[1]);
			delta_ = MatrixXf::Zero(in_dim_[0], in_dim_[1]);
			output_ = MatrixXf::Zero(in_dim_[0], out_dim_[1]);
			std::for_each(state_.begin(), state_.end(), [this](auto &mat) { mat = MatrixXf::Zero(in_dim_[0], out_dim_[1]); });

			// set the shape of each gate and their gradient
			std::for_each(gate_.begin(), gate_.end(), [this](auto &mat) { mat = MatrixXf::Zero(in_dim_[0], out_dim_[1]); });
			std::for_each(dgate_.begin(), dgate_.end(), [this](auto &mat) { mat = MatrixXf::Zero(in_dim_[0], out_dim_[1]); });

			// reload weights
			x_weight_[0] = Eigen::Map <const MatrixXf> (weight[2].data(), in_dim_[1], out_dim_[1]);
			for (int j = 0; j < 2; j ++) { bias_[j] = Eigen::Map <const MatrixXf> (weight[j].data(), 1, out_dim_[1]); }
			for (int j = 3; j < 5; j ++) { h_weight_[j] = Eigen::Map <const MatrixXf> (weight[j].data(), out_dim_[1], out_dim_[1]); }

			thread_pool_.resize(in_dim_[0] <= std::thread::hardware_concurrency() ? in_dim_[0] : in_dim_[0] % std::thread::hardware_concurrency() + 1);
			slope_ = in_dim_[0] / thread_pool_.size();
		}

	private:
		Dim2d in_dim_, out_dim_;																					// input and output shapes 

		std::string recurrent_activation_;																			// names of activation function
  
		MatrixXf input_, delta_, dh_, output_;																// input, layer delta, dJ, output

        std::vector <float> vw_ {2, 1.0}, vh_ {1, 1.0}, vb_ {2, 1.0};												// momentum values

		MatrixXfVec x_weight_, h_weight_, bias_, state_, gate_, dgate_;												// layer's parameters
  
        // empty functions for activation and recurrent activation functions
		std::function <void (MatrixXf &, MatrixXf &, const int &)> activate_, reactivate_;
		
		void time_step_update(MatrixXfVec &db, MatrixXfVec &dxw, MatrixXfVec &dhw, int s, const int end) {
            int e = 0;
			MatrixXf dhp, dzh;

            for (s; s > end; s --) {
				e = randint <int> (-1, s);
				dzh = (dh_.row(s) * h_weight_[1].transpose()).cwiseProduct(dgate_[0].row(s));
				
				db[0] = db[0] + dzh;																			// dbh
                db[1] = db[1] + dh_.row(s);																		// dby
                
				dxw[0] = dxw[0] + input_.col(s) * dzh;															// dWxH(t)
                
                dhw[0] = dhw[0] + state_[0].col(s) * dzh;														// dUhH(t)
                dhw[1] = dhw[1] + gate_[0].col(s) * dh_.row(s);													// dWh

				dhp = dzh * h_weight_[0];

                for (int t = s - 1; t > e; t --) {
					dhp = dhp.cwiseProduct(dgate_[0].row(t));

					db[0] = db[0] + dhp;
					dxw[0] = dxw[0] + input_.col(t) * dhp;
                    dhw[0] = dhw[0] + state_[0].col(t) * dhp;

                    if (t != e + 1) { dhp = dhp * h_weight_[0]; }
                }
            }
        }
};

}
#endif
