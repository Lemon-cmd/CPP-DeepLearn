#ifndef DIST_DENSE_HPP
#define DIST_DENSE_HPP

#include "layer.hpp"

namespace dpp {

class DistributedDense : public Layer {

	/*
	 * Time Distributed Dense Layer 
	 */

	public:
		DistributedDense(const int neurons = 1, const std::string activation = "normal", const float lrate = 0.001, const float erate = 1e-8) {
			/* Save the parameters, and initalize parallelism and set the appropriate activation function */
			assert(neurons > 0 && lrate > 0.0 && erate > 0.0);
			
			// set learning and optimization rate; save neurons to output shape
			lrate_ = lrate;
			erate_ = erate;
			out_dim_[1] = neurons;
			
			// save the activation name for when saving to a file
			activation_ = activation == "normal" ? activation : activation == "relu" ? activation : activation == "tanh" ? activation : activation == "softmax" ? activation : "sigmoid";  
			
			SetActivation(activate_, activation_);	
		}

		void init(const Dim2d in_dim) override {
			/* Initialize the parameters of the layer and save the input shape */
			assert(in_dim[0] > 0 && in_dim[1] > 0);
			// set initialized flag
			init_flag_ = true;

			// save input and output shapes
			in_dim_ = in_dim;
			out_dim_[0] = in_dim[0];

			// initialize weight of bias and weight matrices
			bias_ = MatrixXf::Ones(1, out_dim_[1]);
			weight_ = MatrixXf::Constant(in_dim_[1], out_dim_[1], 2.0f / sqrtf(in_dim_[1] * out_dim_[1]));

			// set the shape of dW, dH, layer delta and layer output
			dh_ = MatrixXf::Zero(in_dim_[0], out_dim_[1]);
			delta_ = MatrixXf::Zero(in_dim_[0], in_dim_[1]);
			output_ = MatrixXf::Zero(in_dim_[0], out_dim_[1]);

			// select the appropriate size of thread pool and calculate the spread in thread executions
			thread_pool_.resize(in_dim_[0] <= std::thread::hardware_concurrency() ? in_dim_[0] : in_dim_[0] % std::thread::hardware_concurrency() + 1);
			slope_ = in_dim_[0] / thread_pool_.size();
		}

		void Topology() {
			std::cout << "\n\t------- Distributed-Dense Layer --------\n"
					  << "\n\tParameters:\t(" << in_dim_[1] * out_dim_[1] + out_dim_[0] * out_dim_[1] << ')'
					  << "\n\tInput-dim:\t(" << in_dim_[0] << ", " << in_dim_[1] << ')'
                      << "\n\tOutput-dim:\t(" << out_dim_[0] << ", " << out_dim_[1] << ")\n"
					  << "\n\t----------------------------------------\n\n";
		}

		void Forward(const MatrixXf &X) override {
			assert(X.rows() == out_dim_[0] && init_flag_);
			/* *
               * Perform H Calculation sequentially
               * Save X
			   * Loop through sequence
			   * Calculate Z and Activate Z
             * */

			input_ = X;
			for (int w = 0; w < out_dim_[0]; w ++)
			{
				output_.row(w) = input_.row(w) * weight_ + bias_;
				activate_(output_, dh_, w);
			}
		}

		void Update() {
			/* *
			   * Transpose weight and save it to another matrix
			   * Transpose input and save it to another matrix
			   * Loop through sequence and perform update
			 * */

			static MatrixXf dw, db;
			dw = MatrixXf::Zero(in_dim_[1], out_dim_[1]);

			input_.transposeInPlace();
			weight_.transposeInPlace();
			
			#pragma omp parallel 
			{
				# pragma omp for nowait
				for (int w = 0; w < out_dim_[0]; w ++) {
					dw = dw + (input_.col(w) * dh_.row(w));						// gradient respects to w
					delta_.row(w) = dh_.row(w) * weight_;						// gradient of layer
				}
			}	

			input_.transposeInPlace();
			weight_.transposeInPlace();
			
			/* average gradients */
			dw = dw.array() / out_dim_[0];
			db = dh_.colwise().sum().array() / out_dim_[0];

			/* calculate weight and bias momentums */
			vw_ = 0.1 * vw_ + 0.9 * dw.array().square().sum();
			vb_ = 0.1 * vb_ + 0.9 * db.array().square().sum();

			/* change bias and weight */
			bias_ = bias_.array() - lrate_ / sqrtf(vb_ + erate_) * db.array();
			weight_ = weight_.array() - lrate_ / sqrtf(vw_ + erate_) * dw.array();

			dw = dw.setZero();
		}

		void SetDelta(const MatrixXf &delta) override {
			// dH = dJ (l + 1) * dH
			dh_ = delta.array() * dh_.array();
		}

		const std::string type() { return "2D"; }
		
		const std::string name() { return "Distributed-Dense"; }
		
		const MatrixXf& Get2dDelta() { return delta_; } 

		const MatrixXf& Get2dOutput() { return output_; }

		const float MeanSquaredError(const MatrixXf &Y, float &accuracy) override {
			// dJ = (H - Y)
			// dH = dJ * dH

			dh_ = (output_.array() - Y.array()) * dh_.array();

			accuracy = output_.unaryExpr<float(*)(float)>(&roundf).isApprox(Y) ? accuracy + 1 : accuracy;

			return 0.5 * (output_ - Y).array().square().sum();
		}

		const float CrossEntropyError(const MatrixXf &Y, float &accuracy) override {
			/*
             * dJ = (-Y / H)
             * dH = dJ * dH = (H - Y)
             */

			// matches of each row
			float hits = 0.0;

			if (activation_ != "softmax") { dh_ = -Y.array() / output_.array() * dh_.array(); }							// set new dJ
			else { dh_ = output_ - Y; }

			// initalize start and end index of thread
			start_ = 0; end_ = slope_;

			// measure accuracy of H concurrently
			for (int t = 0; t < thread_pool_.size(); t ++) {
				thread_pool_[t] = std::thread(&DistributedDense::Measure, this, 
								  std::ref(output_), std::ref(Y), std::ref(hits), start_, end_);

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

			// convert bias and weight matrices to vector
			Eigen::Map<VectorXf> bias_vect (bias_.data(), bias_.size()),
									    weight_vect (weight_.data(), weight_.size());

			// write to file
			file << "\nDistributed Dense\n";
			file << "Activation: " << activation_ << "\n";
			file << "Learning_rate: (" << lrate_ << ")\n";
			file << "Momentum_rate: (" << erate_ << ")\n";
			file << "Input_shape: [" << in_dim_[0] << ',' << in_dim_[1] << "]\n";
			file << "Output_shape: [" << out_dim_[0] << ',' << out_dim_[1] << "]\n";
			file << "Bias: [";
			WriteVector(bias_vect, file);
			file << "Weight: [";
			WriteVector(weight_vect, file);
		}

		void Load(const Dim2d &in_dim, const Dim2d &out_dim, const std::vector <VectorXf> &weight) override {
			assert(weight.size() == 2);

			// set the parameters to metadata inside of the layer
			init_flag_ = true;

			// save input and output shapes
			in_dim_ = in_dim;
			out_dim_ = out_dim_;

			// set the shape of dH, layer delta, and layer output
			dh_ = MatrixXf::Zero(in_dim_[0], out_dim_[1]);
			delta_ = MatrixXf::Zero(in_dim_[0], in_dim_[1]);
			output_ = MatrixXf::Zero(in_dim_[0], out_dim_[1]);

			// map bias and weight vectors to their corresponding matrix
			bias_ = Eigen::Map <const MatrixXf> (weight[0].data(), 1, out_dim_[1]);
			weight_ = Eigen::Map <const MatrixXf> (weight[1].data(), in_dim_[1], out_dim_[1]);

			// set the size of thread pool and calculate the spread in thread executions
			thread_pool_.resize(in_dim_[0] <= std::thread::hardware_concurrency() ? in_dim_[0] : in_dim_[0] % std::thread::hardware_concurrency() + 1);
			slope_ = in_dim_[0] / thread_pool_.size();
		}

	private:
		float vw_ = 0.0, vb_ = 0.0;															// weight and bias momentums
	
		Dim2d in_dim_, out_dim_;															// input and output shapes

		MatrixXf weight_, bias_, input_, output_, delta_, dh_;								// layer's parameters 

		std::function <void (MatrixXf &, MatrixXf &, const int &)> activate_;				// empty activation function
};
}
#endif
