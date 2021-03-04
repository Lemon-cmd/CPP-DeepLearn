#ifndef POOL2D_HPP
#define POOL2D_HPP

namespace dpp {

class Pool2D : public Layer {
	
	/*
	 *
	 * Pooling Layer 2D
	 *
	 */ 

	public:
		Pool2D(const int kernel_height = 1, const int kernel_width = 1, const int height_stride = 1, const int width_stride = 1) {
			assert (kernel_height > 0 && kernel_width > 0 && height_stride > 0 && width_stride > 0);
			/*
             * Save the parameters, and initalize parallelism and set the appropriate activation and recurrent activation functions
             */

			Eigen::initParallel();
			
			// save kernel width and height to ext for slicing
			input_ext_[2] = kernel_width;
			input_ext_[1] = kernel_height;

			// save width and height strided
			width_stride_ = width_stride;
			height_stride_ = height_stride;
		}
		
		void init(const Eigen::DSizes<ptrdiff_t, 3> input_shape) override {
			assert (input_shape[0] > 0 && input_shape[1] > 0 && input_shape[2] > 0 && input_shape[1] > input_ext_[1] && input_shape[2] > input_ext_[2]);
			/*
             *
             * Initialize the parameters of the layer and save the input shape
             *
             */

			// set initialized flag
			init_flag_ = true;	

			// save input
			input_3d_shape_ = input_shape;

			// save channel size
			input_ext_[0] = input_shape[0];
			output_ext_[0] = input_shape[0];

			// calculate mean of summation in pooling; this is used in calculating layer delta
			mean_forward_ = 1.0 / (input_ext_[2] * input_ext_[1]);

			// calculate output shape
			cols_ = (input_shape[1] - input_ext_[2]) / width_stride_ + 1;
			rows_ = (input_shape[0] - input_ext_[1]) / height_stride_ + 1;

			flat_output_size_ = rows_ * cols_ * input_shape[0]; 
			delta_flat_shape_[0] = flat_output_size_;
			
			// set flatten dH and H shape
			flat_output_ = Eigen::Tensor<float, 1> {flat_output_size_};
			flat_doutput_ = Eigen::Tensor<float, 1> {flat_output_size_};
			
			// set shape of H and layer delta
			output_3d_shape_ = Eigen::DSizes<ptrdiff_t, 3> {input_shape[0], rows_, cols_};
			layer_3d_delta_ = Eigen::Tensor<float, 3> {input_shape[0], input_shape[1], input_shape[2]};

			// select the appropriate size of thread pool and calculate the spread in thread executions
			thread_pool_.resize(rows_ <= std::thread::hardware_concurrency() ? rows_ : rows_ % std::thread::hardware_concurrency() + 1);
			slope_ = rows_ / thread_pool_.size();
		}
		
		void topology() {
			std::cout << "\n\t----------- 2D-Pooling Layer -----------\n"
					  << "\n\tStride (Y, X):\t(" << width_stride_ << ", " << height_stride_ << ')'
                      << "\n\tParameters:\t(" << 0 << ')'
					  << "\n\tInput-dim:\t(" << input_3d_shape_[0] << ", " << input_3d_shape_[1] << ", " << input_3d_shape_[2] << ")\n"
                      << "\n\tOutput-dim:\t(" << output_3d_shape_[0] << ", " << output_3d_shape_[1] << ", " << output_3d_shape_[2] << ")\n"
                      << "\n\t----------------------------------------\n\n";
		}

		void forward(const Eigen::Tensor<float, 3> &X) override {
			assert(X.dimension(0) == input_3d_shape_[0] && X.dimension(1) == input_3d_shape_[1] && X.dimension(2) == input_3d_shape_[2] && init_flag_);
			/* Perform Convolution Operation (Concurrently)
             * Create thread pool and perform convolution concurrently
             * Perform convolution on the remainders if exist
             * Map output and its gradients to H and dH
			 */

			// save X
			input_ = X;

			// set index range of thread
			start_ = 0; end_ = slope_;
			
			for (int t = 0; t < thread_pool_.size(); t ++) {
                thread_pool_[t] = std::thread(&Pool2D::ThreadForwardOp, this, start_, end_);
                start_ += slope_; end_ += slope_;
            }
  
            for (int t = 0; t < thread_pool_.size(); t ++) {
                thread_pool_[t].join();
            }
  
            if (rows_ % thread_pool_.size() > 0) {
                ThreadForwardOp(start_, rows_);
            }
			
			// reshape flat H and dH to their intended size
			doutput_ = flat_doutput_.reshape(output_3d_shape_);
			layer_3d_output_ = flat_output_.reshape(output_3d_shape_);
		}

		void update() {
			/*
            * Reset delta
            * Flatten dH
            * Convolve and fill delta
			*/
			
			layer_3d_delta_.setConstant(0.0);

			// flatten new dH
			flat_doutput_ = doutput_.reshape(delta_flat_shape_);

			// reset index range of thread
			start_ = 0; end_ = slope_;

			for (int t = 0; t < thread_pool_.size(); t ++) {
                thread_pool_[t] = std::thread(&Pool2D::ThreadBackwardOp, this, start_, end_);
                start_ += slope_; end_ += slope_;
            }
    
            for (int t = 0; t < thread_pool_.size(); t ++) {
				thread_pool_[t].join();
            }
    
            if (rows_ % thread_pool_.size() > 0) {
                ThreadBackwardOp(start_, rows_);
            }          

		}

		void set_delta(const Eigen::Tensor<float, 3> &prev_layer_delta) override {
			// dH = dJ (l + 1) * dH
            doutput_ = prev_layer_delta * doutput_;
        }
  
		const std::string name() { return "Pool2D"; }

		const std::string type() { return "3D"; } 

        const float MeanSquaredError(const Eigen::Tensor<float, 3> &Y, float &accuracy) override {
			// dH = (H - Y) * dH
            doutput_ = (layer_3d_output_ - Y) * doutput_;

			// convert Y and H into vector for comparison
			Eigen::Map<const Eigen::VectorXf> YH(Y.data(), Y.size());
			Eigen::Map<const Eigen::VectorXf> H(layer_3d_output_.data(), layer_3d_output_.size());

			accuracy = H.isApprox(YH, 0.1) ? accuracy + 1 : accuracy;
            return ((Eigen::Tensor<float, 0>) (0.5 * (layer_3d_output_ - Y).square().sum()))(0);
        }

		void save(std::ofstream &file) {
			assert(init_flag_);

			// write to file
			file << "\nPool2D\n";
			file << "Height_stride: (" << height_stride_ << ")\n";
			file << "Width_stride: (" << width_stride_ << ")\n";
			file << "Input_shape: [" << input_3d_shape_[0] << ',' << input_3d_shape_[1] << ',' << input_3d_shape_[2] << "]\n";
            file << "Output_shape: [" << output_3d_shape_[0] << ',' << output_3d_shape_[1] << ',' << output_3d_shape_[2] << "]\n";
			file << "Kernel_shape: [" << input_ext_[0] << ',' << input_ext_[1] << ',' << input_ext_[2] << "]\n";
		}

		void load(const Eigen::DSizes<ptrdiff_t, 3> &input_shape, const Eigen::DSizes<ptrdiff_t, 3> &output_shape) {
			// set the parameters to metadata inside of the layer
			init_flag_ = true;

			// save input and output shapes
			input_3d_shape_ = input_shape;
			output_3d_shape_ = output_shape;

			// save channel size to ext for slicing
			input_ext_[0] = input_shape[0];
			output_ext_[0] = input_shape[0];

			mean_forward_ = 1.0 / (input_ext_[2] * input_ext_[1]);
			
			// calculate output shape
			cols_ = (input_shape[1] - input_ext_[2]) / width_stride_ + 1;
			rows_ = (input_shape[0] - input_ext_[1]) / height_stride_ + 1;

			flat_output_size_ = rows_ * cols_ * input_shape[0];
			delta_flat_shape_[0] = flat_output_size_;
			
			// set flat dH and H shape
			flat_output_ = Eigen::Tensor<float, 1> {flat_output_size_};
			flat_doutput_ = Eigen::Tensor<float, 1> {flat_output_size_};
			
			// set layer delta shape
			layer_3d_delta_ = Eigen::Tensor<float, 3> {input_shape[0], input_shape[1], input_shape[2]};

			// set the size of thread pool and calculate the spread in thread executions
			thread_pool_.resize(rows_ <= std::thread::hardware_concurrency() ? rows_ : rows_ % std::thread::hardware_concurrency() + 1);
			slope_ = rows_ / thread_pool_.size();
		}	

	private:
		int height_stride_, width_stride_;

		Eigen::Tensor<float, 3> input_, doutput_;
		
		Eigen::Tensor<float, 1> flat_output_, flat_doutput_;

		Eigen::DSizes<ptrdiff_t, 3> input_ext_;
		
		Eigen::DSizes<ptrdiff_t, 1> output_ext_, delta_flat_shape_;

		void ThreadForwardOp(int start, const int end) {
			/* Perform forward convolution
			 * Parameters : start index of rows | end index of rows
			 * Create offsets for output and input slicing
			 * Find maximum value of each channel in input
			 * Size of output's slice is equal to the number of channels
			*/

			int j = start * cols_;
			Eigen::DSizes<ptrdiff_t, 1> output_off {0};

			Eigen::DSizes<ptrdiff_t, 3> input_off {0, 0, 0};

			for (start; start < end; start ++) {
				input_off[1] = start * height_stride_;

				for (int c = 0; c < cols_; c ++) {
					input_off[2] = c * width_stride_;
					
					output_off[0] = j * output_3d_shape_[0];

					flat_output_.slice(output_off, output_ext_) = input_.slice(input_off, input_ext_).maximum(sum_dim_);

					flat_doutput_.slice(output_off, output_ext_) = flat_output_.slice(output_off, output_ext_) * mean_forward_;
					
					j ++;
				}
			}
		}

		void ThreadBackwardOp(int start, const int end) {
			/*
			 * Perform backward convolution
			 * Parameters : start inddex of rows | end index of rows
			 * Create offset for input
			 * Fill in delta
			*/

			int j = start * cols_;
			Eigen::DSizes<ptrdiff_t, 3> input_off {0, 0, 0};

			for (start; start < end; start ++) {
				input_off[1] = start * height_stride_;
				
				for (int c = 0; c < cols_; c ++) {
					input_off[2] = c * width_stride_;
					
					layer_3d_delta_.slice(input_off, input_ext_) = layer_3d_delta_.slice(input_off, input_ext_) + flat_doutput_(j);

					j ++;
				}
			}
		}

};

}

#endif
