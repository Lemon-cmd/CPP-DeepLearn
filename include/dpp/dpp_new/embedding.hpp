#ifndef EMBEDDING_HPP
#define EMBEDDING_HPP

#include "layer.hpp"

namespace dpp {

class Embedding : public Layer {	
	
	/*
	 *
	 *	Word Embedding Layer
	 *  Parameters (lexicon/vocabulary size, neurons, learning rate, optimization rate) 
	 *
	 */

	public:
		Embedding(const int vocab_size = 1, const int neurons = 1, const float lrate = 0.001, const float erate = 1e-8) {
			/* Save the parameters, and initialize Parallelism */
			assert(vocab_size > 0 && neurons > 0 && lrate > 0.0 && erate > 0.0);

			lrate_ = lrate; 
			erate_ = erate;
			out_dim_[1] = neurons;
			vocab_size_ = vocab_size;
		}

		void Topology() {
			std::cout << "\n\t------------ Embedding Layer -----------\n"
					  << "\n\tParameters:\t(" << vocab_size_ * in_dim_[1] * out_dim_[1] << ')'
					  << "\n\tInput-dim:\t(" << in_dim_[0] << ", " << in_dim_[1] << ')'
                      << "\n\tOutput-dim:\t(" << out_dim_[0] << ", " << out_dim_[1] << ")\n"
					  << "\n\t----------------------------------------\n\n";
		}
		
		const std::string type() { return "2D"; }

		const std::string name() { return "Embedding"; }
		
		const MatrixXf& Get2dDelta() override { return delta_; } 

		const MatrixXf& Get2dOutput() override { return output_; }

		const Dim2d Get2dOutputDim() override { return out_dim_; } 

		void init(const Dim2d in_dim) override {
			/* Initialize the parameters of the layer and save the input shape */
			assert(in_dim[0] > 0 && in_dim[1] > 0);
			// set initialized flag
			init_flag_ = true;

			// save input and output shapes
			in_dim_ = in_dim;
			out_dim_[0] = in_dim_[0];

			// initalize weight momentum
			vw_ = std::vector<float> (vocab_size_, 0.0);

			// set shape of delta and output
			delta_ = MatrixXf::Zero(in_dim_[0], in_dim_[1]);
			output_ = MatrixXf::Zero(in_dim_[0], out_dim_[1]);

			// initialize weight
			weight_ = MatrixXfVec (vocab_size_, MatrixXf::Constant(in_dim_[1], out_dim_[1], 2.0f / sqrtf(in_dim_[1] * out_dim_[1])));

			// select the appropriate size of thread pool and calculate the spread in thread executions
			thread_pool_.resize(in_dim_[0] <= std::thread::hardware_concurrency() ? in_dim_[0] : in_dim_[0] % std::thread::hardware_concurrency() + 1);
			slope_ = in_dim_[0] / thread_pool_.size();
		}

		void Forward(const MatrixXf &X) override {
			assert(X.rows() == in_dim_[0] && X.cols() == in_dim_[1] && init_flag_);
			/* *
               * Perform H Calculation concurrently
               * Save X
               * Calculate Z
			 * */

			input_ = X;
			start_ = 0; end_ = slope_;

			for (int t = 0; t < thread_pool_.size(); t ++) {
				thread_pool_[t] = std::thread(&Embedding::ThreadForwardOp, this, start_, end_);
				start_ += slope_; end_ += slope_;
			}

			for (int t = 0; t < thread_pool_.size(); t ++) {
				thread_pool_[t].join();
			}

			if (in_dim_[0] % thread_pool_.size() > 0) {
				ThreadForwardOp(start_, out_dim_[0]);
			}
		}

		void Update() {
			/* *
               * Perform Update concurrently
               * Save current weight
			   * Call Thread Pool
			   * Call Backward Operation
			* */
			
			static MatrixXfVec old_weight;

			old_weight = weight_;

			start_ = 0; end_ = slope_;
			for (int t = 0; t < thread_pool_.size(); t ++) {
                thread_pool_[t] = std::thread(&Embedding::ThreadBackwardOp, this, std::ref(old_weight), start_, end_);
                start_ += slope_; end_ += slope_;
            }

            for (int t = 0; t < thread_pool_.size(); t ++) {
				thread_pool_[t].join();
            }

			if (in_dim_[0] % thread_pool_.size() > 0) {
				ThreadBackwardOp(old_weight, start_, in_dim_[0]);
            }
		}

		void SetDelta(const MatrixXf &delta) override {
			// dL (l + 1) * dH; dH is 1
			dh_ = delta;
		}

		const float MeanSquaredError(const MatrixXf &Y, float &accuracy) override {
			/*
			 * dH = (H - Y) based on derivative of MSquaredError
             * Calculate Accuracy and Loss
            */

			dh_ = output_ - Y;
		   	accuracy = output_.unaryExpr<float(*)(float)>(&roundf).isApprox(Y) ? accuracy + 1 : accuracy;
			return 0.5 * dh_.array().square().sum();
		}

		void Save(std::ofstream &file) {
			assert(init_flag_);
			
			// write to file
			file << "\nEmbedding\n";
			file << "Lexicon_space: (" << vocab_size_ << ")\n";
            file << "Learning_rate: (" << lrate_ << ")\n";
            file << "Momentum_rate: (" << erate_ << ")\n";
			file << "Input_shape: [" << in_dim_[0] << ',' << in_dim_[1] << "]\n";
			file << "Output_shape: [" << out_dim_[0] << ',' << out_dim_[1] << "]\n";
			file << "Weight: [";

			// loop through the row of weight (equivalent to lexicon size)
			for (int j = 0; j < vocab_size_; j ++) {
				// convert weight matrix to vector
				Eigen::Map<VectorXf> weight_vect (weight_[j].data(), weight_[j].size());

				// write the vector to file
				file << '[';

				for (int i = 0; i < weight_vect.size() - 1; i ++) {
					file << weight_vect[i] << ',';
				}

				if (j != vocab_size_ - 1) {
					file << weight_vect[weight_vect.size() - 1] << "],";
				} else {
					file << weight_vect[weight_vect.size() - 1] << ']';
				}
			}

			file << "]\n";
		}

		void Load(const Dim2d &in_dim, const Dim2d &out_dim, const std::vector<VectorXf> &weight) override {
			assert(weight.size() == 1);
			// set the parameters to metadata inside of the layer
			init_flag_ = true;

			// save input and output shapes
			in_dim_ = in_dim;
			out_dim_ = out_dim;

			// set shape of layer delta and output
			delta_ = MatrixXf::Zero(in_dim_[0], in_dim_[1]);
			output_ = MatrixXf::Zero(in_dim_[0], out_dim_[1]);

			// initalize weight momentum
			vw_ = std::vector<float> (vocab_size_, 0.0);

			// resize weight vector to match vocab_size_
			weight_.resize(vocab_size_);

			// unload vectors into weight vector while mapping them into matrix
			for (int j = 0; j < vocab_size_; j ++) {
				weight_[j] = Eigen::Map <const MatrixXf> (weight[j].data(), in_dim_[1], out_dim_[1]);
			}

			// set the size of thread pool and calculate the spread in thread executions
			thread_pool_.resize(in_dim_[0] <= std::thread::hardware_concurrency() ? in_dim_[0] : in_dim_[0] % std::thread::hardware_concurrency() + 1);
			slope_ = in_dim_[0] / thread_pool_.size();
		}

	private:
		int vocab_size_;										// vocabulary size

		MatrixXfVec weight_;									// weight

		Dim2d in_dim_, out_dim_;								// input and output shapes 

		std::vector<float> vw_;									// weight momentum vector

		MatrixXf input_, output_, dh_, delta_;					// input and dH matrices

		/*
		 *
		 * Thread Forward Operation
		 * Perform H calculation given an appropriate index
		 *
		*/

		void ThreadForwardOp(int s, const int e) {
			int index;
			for (s; s < e; s ++)
			{
				// if input is between 0 and lexicon size, then it is the index. Else, index is 0.
				index = input_.row(s).sum();
				index = index < vocab_size_ && index >= 0 ? index : index > vocab_size_ ? vocab_size_ - 1 : 0;

				// calculate H in its corresponding row
				output_.row(s) = input_.row(s) * weight_[index];
			}
		}

		void ThreadBackwardOp(MatrixXfVec &old_weight, int s, const int e) {
			int index;
			MatrixXf dw;		

			for (s; s < e; s ++)
			{
				// set index
				index = input_.row(s).sum();
				index = index < vocab_size_ && index >= 0 ? index : index > vocab_size_ ? vocab_size_ - 1 : 0;

				// calculate dW
				dw = input_.row(s).transpose() * dh_.row(s);

				// calculate gradient of layer in its corresponding row
				delta_.row(s) = dh_.row(s) * old_weight[index].transpose();

				// Lock and make the appropriate changes to weight momentum and weight. Then, unlock.
				thread_lock_.lock();
				
				vw_[index] = 0.1 * vw_[index] + 0.9 * dw.array().square().sum();
				weight_[index] = weight_[index].array() - lrate_ / sqrtf(vw_[index] + erate_) * dw.array();

				thread_lock_.unlock();
			}
		}

};
}
#endif

