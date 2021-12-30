#include "layer.hpp"

namespace dpp {

class Dense : public Layer {
	/*
	 * Feed Forward Layer
	 */

	public:
		Dense(const int neurons = 1, const std::string activation = "normal", const float lrate = 0.001, const float erate = 1e-8) {
			/* Save the parameters, and initalize parallelism and set the appropriate activation function */
			assert(neurons > 0 && lrate > 0.0 & erate > 0.0);

			out_dim_[1] = neurons;

			lrate_ = lrate; erate_ = erate;
			
			activation_ = activation == "normal" ? activation : activation == "tanh" ? activation \
						: activation == "softmax" ? activation : activation == "relu" ? activation : "sigmoid"; 
		
			// set activation function	
			SetActivation(activate_, activation_);
		}

		const std::string type() { return "2D"; }
	
		const std::string name() { return "Dense"; }

		const MatrixXf& Get2dDelta() override { return delta_; } 

		const MatrixXf& Get2dOutput() override { return output_; }

		const Dim2d Get2dOutputDim() override { return out_dim_; } 

		void Topology() { 
			std::cout << "\n\t-------------- Dense Layer -------------\n"
					  << "\n\tParameters:\t(" << in_dim_[1] * out_dim_[1] + out_dim_[0] * out_dim_[1] << ')'
					  << "\n\tInput-dim:\t(" << in_dim_[0] << ", " << in_dim_[1] << ')'
                      << "\n\tOutput-dim:\t(" << out_dim_[0] << ", " << out_dim_[1] << ")\n"
					  << "\n\t----------------------------------------\n\n";
		}

		void init(const Dim2d in_dim) override {
			/* Initialize the parameters of the layer and save the input shape */
			assert(in_dim[0] > 0 && in_dim[1] > 0);
			// set initialized flag
			init_flag_ = true;

			// save output shape
			in_dim_ = in_dim;
			out_dim_[0] = in_dim_[0];

			// initialize weights of bias and weight matrices
			bias_ = MatrixXf::Ones(in_dim_[0], out_dim_[1]);
			weight_ = MatrixXf::Constant(in_dim_[1], out_dim_[1], 2.0f / sqrtf(in_dim_[1] * out_dim_[1]));

			// select the appropriate size of thread pool and calculate the spread in thread executions
			thread_pool_.resize(in_dim_[0] <= std::thread::hardware_concurrency() ? in_dim_[0] : in_dim_[0] % std::thread::hardware_concurrency() + 1);
			slope_ = in_dim_[0] / thread_pool_.size();
		}

		void Forward(const MatrixXf &X) override {
			assert(X.rows() == in_dim_[0] && X.cols() == in_dim_[1] && init_flag_);

			input_ = X;									// save X
			output_ = input_ * weight_ + bias_;			// calculate Z
			activate_(output_, dh_);					// activate and calculate dg(Z)
		}

		void Update() {
			static MatrixXf dw;
			// calculate gradient of weight
			dw = input_.transpose() * dh_;

			// calculate the momentums of bias and weights based on Adam prop.
			vw_ = 0.1 * vw_ + 0.9 * dw.array().square().sum();
			vb_ = 0.1 * vb_ + 0.9 * dh_.array().square().sum();

			// change bias
			bias_ = bias_.array() - lrate_ / sqrtf(vb_ + erate_) * dh_.array();

			// calculate the gradient of the layer before changing its weight
			delta_ = dh_ * weight_.transpose();

			// change weight
			weight_ = weight_.array() - lrate_ / sqrtf(vw_ + erate_) * dw.array();
		}

		void SetDelta(const MatrixXf &delta) override {
			dh_ = dh_.cwiseProduct(delta);	
		}

		const float MeanSquaredError(const MatrixXf &Y, float &accuracy) override {
			/* 
			 * dL = (H - Y) based on derivative of MSquaredError
			 * dH = dL * dH 
			 * Calculate Accuracy and Loss
			*/
			
			dh_ = (output_ - Y).array() * dh_.array();

		   	accuracy = output_.unaryExpr<float(*)(float)>(&roundf).isApprox(Y) ? accuracy + 1 : accuracy;

			return 0.5 * (output_ - Y).array().square().sum();
		}

		const float CrossEntropyError(const MatrixXf &Y, float &accuracy) override {
			/*
			 * dH = -Y / H
			 * dL * dH = (-Y / H) * (H * (1 - H)) = H - Y
			 * Calculate Accuracy and Loss
			*/

			static float hits; hits = 0.0;
			
			if (activation_ != "softmax") { dh_ = -Y.array() / output_.array() * dh_.array(); }							// set new dJ
			else { dh_ = output_ - Y; }
		
			// initalize start and end index of thread
			start_ = 0; end_ = slope_;

			// measure accuracy of H concurrently
			for (int t = 0; t < thread_pool_.size(); t ++) {
				thread_pool_[t] = std::thread(&Dense::Measure, this, std::ref(output_), std::ref(Y), std::ref(hits), start_, end_);
				start_ += slope_; end_ += slope_;
			}

			for (int t = 0; t < thread_pool_.size(); t ++) {
				thread_pool_[t].join();
			}

			// perform calculation on the remainder if exists
			if (out_dim_[0] % thread_pool_.size() > 0) {
				Measure(output_, Y, hits, start_, out_dim_[0]);
			}

			// accuracy = average of matches argmax of H to Y
			accuracy = hits / out_dim_[0] + accuracy;

			return (1.0 - Y.array() * output_.array().log()).sum();
		}

		void Save(std::ofstream &file) {
			assert(init_flag_);

			// convert the bias and weight into vectors
			Eigen::Map<VectorXf> bias_vect (bias_.data(), bias_.size()), weight_vect (weight_.data(), weight_.size());
			
			// write to file 
			file << "\nDense\n";
			file << "Activation: " << activation_ << '\n';
			file << "Learning_rate: (" << lrate_ << ")\n";
			file << "Momentum_rate: (" << erate_ << ")\n";
			file << "Input_shape: [" << in_dim_[0] << ',' << in_dim_[1] << "]\n";
			file << "Output_shape: [" << out_dim_[0] << ',' << out_dim_[1] << "]\n";
			file << "Bias: [";
			WriteVector(bias_vect, file);	
			file << "Weight: [";	
			WriteVector(weight_vect, file);
		}

		void Load(const Dim2d &in_dim, const Dim2d &out_dim, const std::vector<VectorXf> &weight) override {
			assert(weight.size() == 2);
			// set the parameters to metadata inside of the layer	
			init_flag_ = true;
			
			// save input and output shapes
			in_dim_ = in_dim; out_dim_ = out_dim;

			// map the vectors to their corresponding matrix
			bias_ = Eigen::Map <const MatrixXf> (weight[0].data(), out_dim_[0], out_dim_[1]);
			weight_ = Eigen::Map <const MatrixXf> (weight[1].data(), in_dim_[1], out_dim_[1]);
			
			// set the size of thread pool and calculate the spread in thread executions
			thread_pool_.resize(in_dim_[0] <= std::thread::hardware_concurrency() ? in_dim_[0] : in_dim_[0] % std::thread::hardware_concurrency() + 1);
			slope_ = in_dim_[0] / thread_pool_.size();
		}

	private:
		float vw_ = 0.0, vb_ = 0.0;															// momentums of bias and weight
		Dim2d in_dim_, out_dim_;															// input and output shapes 
		MatrixXf input_, weight_, bias_, output_, dh_, delta_;								// input, weight and bias
		std::function <void (MatrixXf &, MatrixXf &)> activate_;							// activation function
};

}
