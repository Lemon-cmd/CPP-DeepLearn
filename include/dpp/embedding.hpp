#ifndef EMBEDDING_HPP
#define EMBEDDING_HPP

namespace dpp {

class Embedding : public Layer {	
	
	/*
	 *
	 *	Word Embedding Layer
	 *  Parameters (lexicon/vocabulary size, neurons, learning rate, optimization rate) 
	 *
	 */

	public:
		Embedding(const int lexicon_size = 1, const int neurons = 1, const float lrate = 0.001, const float erate = 1e-8) {
			assert(lexicon_size > 0 && neurons > 0 && lrate > 0.0 && erate > 0.0);
			/* * 
			 *
			 * Save the parameters, and initialize Parallelism 
			 *
			 * */ 

			Eigen::initParallel();	
			lrate_ = lrate; 
			erate_ = erate;
			lexicon_size_ = lexicon_size;
			output_2d_shape_[1] = neurons;
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
			output_2d_shape_[0] = input_2d_shape_[0];
			
			// initalize weight momentum
			vW_ = std::vector<float> (lexicon_size_, 0.0);

			// set shape of delta and output
			layer_2d_delta_ = Eigen::MatrixXf::Zero(input_shape[0], input_shape[1]);
			layer_2d_output_ = Eigen::MatrixXf::Zero(input_shape[0], output_2d_shape_[1]);
			
			// initialize weight
			weight_ = std::vector<Eigen::MatrixXf> (lexicon_size_, Eigen::MatrixXf::Constant(input_shape[1], output_2d_shape_[1], 2.0f / sqrtf(input_shape[1] * output_2d_shape_[1])));

			// select the appropriate size of thread pool and calculate the spread in thread executions
			thread_pool_.resize(input_shape[0] <= std::thread::hardware_concurrency() ? input_shape[0] : input_shape[0] % std::thread::hardware_concurrency() + 1);
			slope_ = input_shape[0] / thread_pool_.size();
		}
		
		void topology() {
			std::cout << "\n\t------------ Embedding Layer -----------\n"
					  << "\n\tParameters:\t(" << lexicon_size_ * input_2d_shape_[1] * output_2d_shape_[1] << ')'
					  << "\n\tInput-dim:\t(" << input_2d_shape_[0] << ", " << input_2d_shape_[1] << ')'
                      << "\n\tOutput-dim:\t(" << output_2d_shape_[0] << ", " << output_2d_shape_[1] << ")\n"
					  << "\n\t----------------------------------------\n\n";
		}
		
		void forward(const Eigen::MatrixXf &X) override {
			assert(X.rows() == input_2d_shape_[0] && X.cols() == input_2d_shape_[1] && init_flag_);
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

			if (input_2d_shape_[0] % thread_pool_.size() > 0) {
				ThreadForwardOp(start_, output_2d_shape_[0]);
			}
		}

		void update() {
			/* *
               * Perform Update concurrently 
               * Save current weight
			   * Call Thread Pool
			   * Call Backward Operation
			* */
			
			old_weight_ = weight_;
			
			start_ = 0; end_ = slope_;
            
			for (int t = 0; t < thread_pool_.size(); t ++) {
                thread_pool_[t] = std::thread(&Embedding::ThreadBackwardOp, this, start_, end_);
                start_ += slope_; end_ += slope_;
            }

            for (int t = 0; t < thread_pool_.size(); t ++) {
				thread_pool_[t].join();
            }
            
			if (input_2d_shape_[0] % thread_pool_.size() > 0) {
				ThreadBackwardOp(start_, input_2d_shape_[0]);
            }
		}

		void set_delta(const Eigen::MatrixXf &prev_layer_delta) override {
			// dL (l + 1) * dH; dH is 1
			doutput_ = prev_layer_delta;
		}

		const std::string name() { return "Embedding"; }

		const std::string type() { return "2D"; }

		const float MeanSquaredError(const Eigen::MatrixXf &Y, float &accuracy) override {
			/*
			 * dH = (H - Y) based on derivative of MSquaredError
             * Calculate Accuracy and Loss
            */

			doutput_ = layer_2d_output_ - Y;
		   	accuracy = layer_2d_output_.unaryExpr<float(*)(float)>(&roundf).isApprox(Y) ? accuracy + 1 : accuracy;
			return 0.5 * doutput_.array().square().sum();
		}

		const float CrossEntropyError(const Eigen::MatrixXf &Y, float &accuracy) override {
			/*
			 * dJ = (-Y / H)
			 * dH = dJ * dH = dJ, where dH = 1
			*/

			// matches of each row
			float hits = 0.0;
			
			// set new dH
			doutput_ = -Y.array() / layer_2d_output_.array();
			
			// initalize start and end index of thread
			start_ = 0; end_ = slope_;
			
			// measure accuracy of H concurrently
			for (int t = 0; t < thread_pool_.size(); t ++) {
				thread_pool_[t] = std::thread(&Embedding::measure2d, this, std::ref(layer_2d_output_), std::ref(Y), std::ref(hits), start_, end_);
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

			// write to file 
			file << "\nEmbedding\n";
			file << "Lexicon_space: (" << lexicon_size_ << ")\n";
            file << "Learning_rate: (" << lrate_ << ")\n";
            file << "Momentum_rate: (" << erate_ << ")\n";
			file << "Input_shape: [" << input_2d_shape_[0] << ',' << input_2d_shape_[1] << "]\n";
			file << "Output_shape: [" << output_2d_shape_[0] << ',' << output_2d_shape_[1] << "]\n";
			file << "Weight: [";
			
			// loop through the row of weights (equivalent to lexicon size)
			for (int j = 0; j < lexicon_size_; j ++) {	
				// convert weight matrix to vector
				Eigen::Map<Eigen::VectorXf> weight_vect (weight_[j].data(), weight_[j].size());
				
				// write the vector to file
				file << '[';

				for (int i = 0; i < weight_vect.size() - 1; i ++) {
					file << weight_vect[i] << ',';
				}
				
				if (j != lexicon_size_ - 1) {
					file << weight_vect[weight_vect.size() - 1] << "],";
				} else {
					file << weight_vect[weight_vect.size() - 1] << ']';
				}
			}

			file << "]\n";
		}	
	
		void load(const Eigen::DSizes<ptrdiff_t, 2> &input_shape, 
				  const Eigen::DSizes<ptrdiff_t, 2> &output_shape,
				  const std::vector<std::vector<float>> &weight) override {
			
			// set the parameters to metadata inside of the layer
			init_flag_ = true;

			// save input and output shapes
			input_2d_shape_ = input_shape; 
			output_2d_shape_ = output_shape;

			// initalize weight momentum
			vW_ = std::vector<float> (lexicon_size_, 0.0);

			// resize weight vector to match lexicon_size_
			weight_.resize(lexicon_size_);

			// unload vectors into weight vector while mapping them into matrix
			for (int j = 0; j < lexicon_size_; j ++) {
				weight_[j] = Eigen::Map<const Eigen::MatrixXf> (weight[j].data(), input_shape[1], output_shape[1]); 
			}
			
			// set shape of layer delta and output
			layer_2d_delta_ = Eigen::MatrixXf::Zero(input_shape[0], input_shape[1]);
			layer_2d_output_ = Eigen::MatrixXf::Zero(input_shape[0], output_shape[1]);

			// set the size of thread pool and calculate the spread in thread executions
			thread_pool_.resize(input_shape[0] <= std::thread::hardware_concurrency() ? input_shape[0] : input_shape[0] % std::thread::hardware_concurrency() + 1);
			slope_ = input_shape[0] / thread_pool_.size();
		}

	private:
		int lexicon_size_;										// vocabulary size

		Eigen::MatrixXf input_, doutput_;						// input and dH matrices

		std::vector<float> vW_;									// weight momentum vector
		
		std::vector<Eigen::MatrixXf> weight_, old_weight_;		// weight and old_weight vectors

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
				index = index < lexicon_size_ && index >= 0 ? index : index > lexicon_size_ ? lexicon_size_ - 1 : 0;

				// calculate H in its corresponding row
				layer_2d_output_.row(s) = (input_.row(s) * weight_[index]).eval();
			}
		}

		void ThreadBackwardOp(int s, const int e) {
			int index;
			Eigen::MatrixXf dweight_;		// dW
			
			for (s; s < e; s ++)
			{
				// set index
				index = input_.row(s).sum();
				index = index < lexicon_size_ && index >= 0 ? index : index > lexicon_size_ ? lexicon_size_ - 1 : 0;

				// calculate dW
				dweight_ = input_.row(s).transpose() * doutput_.row(s);

				// calculate gradient of layer in its corresponding row
				layer_2d_delta_.row(s) = doutput_.row(s) * old_weight_[index].transpose(); 

				// Lock and make the appropriate changes to weight momentum and weight. Then, unlock.
				thread_lock_.lock();
				
				vW_[index] = 0.1 * vW_[index] + 0.9 * dweight_.array().square().sum();
				weight_[index] = weight_[index].array() - lrate_ / sqrtf(vW_[index] + erate_) * dweight_.array();
				
				thread_lock_.unlock();
			}
		}
};

}

#endif
