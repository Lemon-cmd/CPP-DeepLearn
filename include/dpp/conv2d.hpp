#ifndef CONV2D_HPP
#define CONV2D_HPP

namespace dpp {

class Conv2D : public Layer {

	/* 
	 *
	 * Convolutional Layer 2D
	 *
	 */
	
	public:
		Conv2D (const int num_kernels = 1, const int kernel_height = 1, const int kernel_width = 1, const int height_stride = 1, const int width_stride = 1, 
			    const float lrate = 0.001, const std::string activation = "normal", const float erate = 1e-8) {

			assert (num_kernels > 0 && kernel_height > 0 && kernel_width > 0 && height_stride > 0 && width_stride > 0 && lrate > 0.0 && erate > 0.0);	
			/*
             * Save the parameters, and initalize parallelism and set the appropriate activation and recurrent activation functions
            */

			Eigen::initParallel();
			
			// set learning and optimization rate
			lrate_ = lrate;
			erate_ = erate;
			
			// set ext of kernel and input slice
			input_ext_[2] = kernel_width;
			input_ext_[1] = kernel_height;
			
			kernel_ext_[0] = 1;
			kernel_ext_[3] = kernel_width;
			kernel_ext_[2] = kernel_height;

			// set stride and save kernel size to output
			width_stride_ = width_stride;
			height_stride_ = height_stride;
			output_3d_shape_[0] = num_kernels;

			// bind activation function first to normal function aka no activation
			act_func_ = std::bind(&Conv2D::conv_normal, this, std::placeholders::_1,
								  std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5, std::placeholders::_6,
								  std::placeholders::_7, std::placeholders::_8, std::placeholders::_9, std::placeholders::_10, std::placeholders::_11);

			// save the activation name for when saving to a file
			activation_ = activation == "normal" ? activation : activation == "tanh" ? activation : activation == "softmax" ? activation : activation == "relu" ? activation : "sigmoid"; 

			// make any necessary changes to activation function
			if (activation == "relu") {
				act_func_ = std::bind(&Conv2D::conv_relu, this, std::placeholders::_1,
									  std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5, std::placeholders::_6,
                                      std::placeholders::_7, std::placeholders::_8, std::placeholders::_9, std::placeholders::_10, std::placeholders::_11);
			} else if (activation == "tanh") {
				act_func_ = std::bind(&Conv2D::conv_tanh, this, std::placeholders::_1,
                                      std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5, std::placeholders::_6,
                                      std::placeholders::_7, std::placeholders::_8, std::placeholders::_9, std::placeholders::_10, std::placeholders::_11);
			} else if (activation == "sigmoid") {
			    act_func_ = std::bind(&Conv2D::conv_sigmoid, this, std::placeholders::_1,
                                      std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5, std::placeholders::_6,
									  std::placeholders::_7, std::placeholders::_8, std::placeholders::_9, std::placeholders::_10, std::placeholders::_11);
			} else if (activation == "softmax") {
				act_func_ = std::bind(&Conv2D::conv_softmax, this, std::placeholders::_1,
                                      std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5, std::placeholders::_6,
                                      std::placeholders::_7, std::placeholders::_8, std::placeholders::_9, std::placeholders::_10, std::placeholders::_11);
			}
		}

		void init(const Eigen::DSizes<ptrdiff_t, 3> input_shape) override {
			assert(input_shape[0] > 0 && input_shape[1] > 0 && input_shape[2] > 0 && input_shape[1] >= kernel_ext_[2] && input_shape[2] >= kernel_ext_[3]);
			/*
             *
             * Initialize the parameters of the layer and save the input shape
             *
            */

			// set initialized flag
			init_flag_ = true;

			// save input and calculate output shape
			input_3d_shape_ = input_shape;
			rows_ = (input_shape[1] - kernel_ext_[2]) / height_stride_ + 1;
			cols_ = (input_shape[2] - kernel_ext_[3]) / width_stride_ + 1;

			output_3d_shape_[1] = rows_;
		    output_3d_shape_[2] = cols_;

			// save channel size of input
			input_ext_[0] = input_shape[0];
			kernel_ext_[1] = input_shape[0];

			// mean of summation in convolution
			mean_forward_ = 1.0 / (input_shape[0] * kernel_ext_[2] * kernel_ext_[3]);

			// total size of output
            flat_output_size_ = rows_ * cols_ * output_3d_shape_[0];
			delta_flat_shape_[0] = flat_output_size_;

			flat_output_ = Eigen::Tensor<float, 1> {flat_output_size_};
            flat_doutput_ = Eigen::Tensor<float, 1> {flat_output_size_};

			// momentum of weight and bias 
			vW_ = Eigen::Tensor<float, 1> {output_3d_shape_[0]};
			vB_ = Eigen::Tensor<float, 1> {output_3d_shape_[0]};

			// set shape of bias, its gradient and weight gradient matrices
			bias_ = Eigen::Tensor<float, 1> {output_3d_shape_[0]};
			dbias_ = Eigen::Tensor<float, 1> {output_3d_shape_[0]};
			dweight_ = Eigen::Tensor<float, 1> {output_3d_shape_[0]};

			// set shape of relu matrices used for comparison (min and max functions)
			relu_3d_ones_ = Eigen::Tensor<float, 3> {input_shape[0], kernel_ext_[2], kernel_ext_[3]};
			relu_3d_zeroes_ = Eigen::Tensor<float, 3> {input_shape[0], kernel_ext_[2], kernel_ext_[3]};

			// set delta kernel shape
            layer_3d_delta_ = Eigen::Tensor<float, 3> {input_shape[0], input_shape[1], input_shape[2]};
			kernel_ = Eigen::Tensor<float, 4> {output_3d_shape_[0], input_shape[0], kernel_ext_[2], kernel_ext_[3]};

			/* comparison tensors for relu function */
			relu_3d_ones_.setConstant(1.0);
			relu_3d_zeroes_.setConstant(0.0);

			/* set values of weights and momentums */
			vB_.setConstant(0.0);
            vW_.setConstant(0.0);
			bias_.setConstant(1.0);
			dweight_.setConstant(0.0);
			kernel_.setConstant(pow(output_3d_shape_[0] * input_shape[0] * kernel_ext_[2] * kernel_ext_[3], 1.0 / 4.0));
			
			// select the appropriate size of thread pool and calculate the spread in thread executions
			thread_pool_.resize(output_3d_shape_[0] <= std::thread::hardware_concurrency() ? output_3d_shape_[0] : output_3d_shape_[0] % std::thread::hardware_concurrency() + 1);
			slope_ = output_3d_shape_[0] / thread_pool_.size();
		}

		void topology() {
			std::cout << "\n\t-------- 2D-Convolutional Layer --------\n"
					  << "\n\tStride (Y, X):\t(" << height_stride_ << ", " << width_stride_ << ')'
					  << "\n\tParameters:\t(" << output_3d_shape_[0] + output_3d_shape_[0] * kernel_ext_[1] * kernel_ext_[2] * kernel_ext_[3] << ')'
                      << "\n\tInput-dim:\t(" << input_3d_shape_[0] << ", " << input_3d_shape_[1] << ", " << input_3d_shape_[2] << ')'
                      << "\n\tOutput-dim:\t(" << output_3d_shape_[0] << ", " << output_3d_shape_[1] << ", " << output_3d_shape_[2] << ")\n"
                      << "\n\t----------------------------------------\n\n";
		}

		void forward(const Eigen::Tensor<float, 3> &X) override {
			assert(X.dimension(0) == input_3d_shape_[0] && X.dimension(1) == input_3d_shape_[1] && X.dimension(2) == input_3d_shape_[2] && init_flag_);
			/* Perform Convolution Operation (Concurrently)
              * Create thread pool and perform convolution concurrently
              * Perform convolution if there exists remainders
              * Else, convolve serially
              * Map output and its gradients to H and dH
            */

			input_ = X;

			start_ = 0; end_ = slope_;

			for (int t = 0; t < thread_pool_.size(); t ++) {
				thread_pool_[t] = std::thread(&Conv2D::ThreadForwardOp, this, start_, end_);
				start_ += slope_; end_ += slope_;
			}

			for (int t = 0; t < thread_pool_.size(); t ++) {
				thread_pool_[t].join();
			}

			if (output_3d_shape_[0] % thread_pool_.size() > 0) {
				ThreadForwardOp(start_, output_3d_shape_[0]);
			}
			
			// reshape dH and H
			doutput_ = flat_doutput_.reshape(output_3d_shape_);
			layer_3d_output_ = flat_output_.reshape(output_3d_shape_);
		}

		void update() {
			/* Reset delta
			* Update B
			* Reshape dH to 1D matrix
			* Update K and set new delta to pass
			*/

			layer_3d_delta_.setConstant(0.0);

			// calculate sum of dH over axis 1 and 2; shape becomes {num_of_filters, 1}
			dbias_ = doutput_.sum(sum_dim_);
			
			// momentum of each filter
			vB_ = 0.1 * vB_ + 0.9 * dbias_.square();
			
			// flatten new dH
			flat_doutput_ = doutput_.reshape(delta_flat_shape_);
			
			// change bias
			bias_ = bias_ - vB_.setConstant(lrate_) / (vB_ + erate_).sqrt() * dbias_;

			start_ = 0; end_ = slope_;

			for (int t = 0; t < thread_pool_.size(); t ++) {
                thread_pool_[t] = std::thread(&Conv2D::ThreadBackwardOp, this, start_, end_);
                start_ += slope_; end_ += slope_;
            }

            for (int t = 0; t < thread_pool_.size(); t ++) {
				thread_pool_[t].join();
            }

            if (output_3d_shape_[0] % thread_pool_.size() > 0) {
                ThreadBackwardOp(start_, output_3d_shape_[0]);
            }

			dweight_.setConstant(0.0);
		}

		void set_delta(const Eigen::Tensor<float, 3> &prev_layer_delta) override {
			// dH = dJ (l + 1) * dH
			doutput_ = prev_layer_delta * doutput_;
		}

		const std::string name() { return "Conv2D"; }

		const std::string type() { return "3D"; }

		const float MeanSquaredError(const Eigen::Tensor<float, 3> &Y, float &accuracy) override {
			// dH = (H - Y) * dH
			doutput_ = (layer_3d_output_ - Y) * doutput_;

			// convert H and YH into vector for comparison
			Eigen::Map<const Eigen::VectorXf> YH(Y.data(), Y.size());
			Eigen::Map<const Eigen::VectorXf> H(layer_3d_output_.data(), layer_3d_output_.size());

			accuracy = H.isApprox(YH, 0.1) ? accuracy + 1 : accuracy;
			return ((Eigen::Tensor<float, 0>) (0.5 * (layer_3d_output_ - Y).square().sum()))(0);
		}
		
		void save(std::ofstream &file) {
			assert(init_flag_);

			// convert kernel and bias into vector
			Eigen::Map<Eigen::VectorXf> kernel_vect (kernel_.data(), kernel_.size()), bias_vect (bias_.data(), bias_.size());

			// write to file
			file << "\nConv2D\n";
			file << "Activation: " << activation_ << '\n';				
			file << "Learning_rate: (" << lrate_ << ")\n";
			file << "Momentum_rate: (" << erate_ << ")\n";
			file << "Height_stride: (" << height_stride_ << ")\n";
			file << "Width_stride: (" << width_stride_ << ")\n";
			file << "Input_shape: [" << input_3d_shape_[0] << ',' << input_3d_shape_[1] << ',' << input_3d_shape_[2] << "]\n";
			file << "Output_shape: [" << output_3d_shape_[0] << ',' << output_3d_shape_[1] << ',' << output_3d_shape_[2] << "]\n";
			file << "Kernel_shape: [" << output_3d_shape_[0] << ',' << kernel_ext_[1] << ',' << kernel_ext_[2] << ',' << kernel_ext_[3] << "]\n";
			file << "Bias: [";
			write_vector(bias_vect, file);
			file << "Weight: [";
			write_vector(kernel_vect, file);
		}

		void load(const Eigen::DSizes<ptrdiff_t, 3> &input_shape,
				  const Eigen::DSizes<ptrdiff_t, 3> &output_shape,
				  std::vector<float> &bias,
				  std::vector<float> &weight) override {
			
			// set the parameters to metadata inside of the layer
			init_flag_ = true;

			// save input and output shapes
			input_3d_shape_ = input_shape;
			output_3d_shape_ = output_shape;
			
			rows_ = output_3d_shape_[1]; 
			cols_ = output_3d_shape_[2];

			// save channel size
			input_ext_[0] = input_shape[0];
            kernel_ext_[1] = input_shape[0];
			
			// calculate total size of H and dH
			flat_output_size_ = rows_ * cols_ * output_3d_shape_[0];

			// mean of summation in convolution 
			mean_forward_ = 1.0 / (input_shape[0] * kernel_ext_[2] * kernel_ext_[3]);
			delta_flat_shape_[0] = flat_output_size_;

			// set shape of dH and H
			flat_output_ = Eigen::Tensor<float, 1> {flat_output_size_};
            flat_doutput_ = Eigen::Tensor<float, 1> {flat_output_size_};
			
			// set momentums of bias and weight shape
			vW_ = Eigen::Tensor<float, 1> {output_3d_shape_[0]};
			vB_ = Eigen::Tensor<float, 1> {output_3d_shape_[0]};

			// set gradients of bias and weight shape
			dbias_ = Eigen::Tensor<float, 1> {output_3d_shape_[0]};
			dweight_ = Eigen::Tensor<float, 1> {output_3d_shape_[0]};

			// set relu comparison matrices' shape
			relu_3d_ones_ = Eigen::Tensor<float, 3> {input_shape[0], kernel_ext_[2], kernel_ext_[3]};
			relu_3d_zeroes_ = Eigen::Tensor<float, 3> {input_shape[0], kernel_ext_[2], kernel_ext_[3]};

			// set layer delta shape
			layer_3d_delta_ = Eigen::Tensor<float, 3> {input_shape[0], input_shape[1], input_shape[2]};
			
			// map the vectors to their corresponding matrix
			bias_ = Eigen::TensorMap<Eigen::Tensor<float, 1>> (bias.data(), Eigen::DSizes<ptrdiff_t, 1> {output_3d_shape_[0]});
			kernel_ = Eigen::TensorMap<Eigen::Tensor<float, 4>> (weight.data(), Eigen::DSizes<ptrdiff_t, 4> {output_3d_shape_[0], input_shape[0], kernel_ext_[2], kernel_ext_[3]});

			// set values of matrices
			relu_3d_ones_.setConstant(1.0);
			relu_3d_zeroes_.setConstant(0.0);

			vB_.setConstant(0.0);
            vW_.setConstant(0.0);
			dweight_.setConstant(0.0);

			// set the size of thread pool and calculate the spread in thread executions
			thread_pool_.resize(output_3d_shape_[0] <= std::thread::hardware_concurrency() ? output_3d_shape_[0] : output_3d_shape_[0] % std::thread::hardware_concurrency() + 1);
			slope_ = output_3d_shape_[0] / thread_pool_.size();
		}
	
	private:
		
		std::string activation_;

		int width_stride_, height_stride_;
		
		Eigen::Tensor<float, 4> kernel_;
		
		Eigen::Tensor<float, 3> input_, doutput_;
		
		Eigen::Tensor<float, 1> bias_, dweight_, dbias_, vW_, vB_, flat_output_, flat_doutput_;

		// ext matrices are used for slicing 
		Eigen::DSizes<ptrdiff_t, 3> input_ext_;
		
		Eigen::DSizes<ptrdiff_t, 4> kernel_ext_;

		Eigen::DSizes<ptrdiff_t, 1> delta_flat_shape_;

		std::function <void (const Eigen::Tensor<float, 3> &,
							 const Eigen::Tensor<float, 4> &,
							 const Eigen::Tensor<float, 1> &,
							 Eigen::Tensor<float, 1> &,
							 Eigen::Tensor<float, 1> &,
							 const Eigen::DSizes<ptrdiff_t, 3> &,
							 const Eigen::DSizes<ptrdiff_t, 3> &,
							 const Eigen::DSizes<ptrdiff_t, 4> &,
							 const Eigen::DSizes<ptrdiff_t, 4> &,
							 const int &, const int &)> act_func_;

		void ThreadForwardOp(int start, const int end) {
			/* Parameters : starting filter index | ending filter index
			 * Create offsets for slicing tensors
			 * Convolve forward and fill in O and dO which are later reshaped into H and dH
			 * Set appropriate values for X-offset and K-offset
			 * No mutex is needed as operations do not need to wait
			*/

            Eigen::DSizes<ptrdiff_t, 3> input_off {0, 0, 0};
			Eigen::DSizes<ptrdiff_t, 4> kernel_off {0, 0, 0, 0};
			int j = start * rows_ * cols_;

			for (start; start < end; start ++) {
				kernel_off[0] = start;
                for (int r = 0; r < rows_; r ++) {
                    input_off[1] = r * height_stride_;
                    for (int c = 0; c < cols_; c ++) {
                        input_off[2] = c * width_stride_;
                        act_func_(input_, kernel_, bias_, flat_output_, flat_doutput_, input_off, input_ext_, kernel_off, kernel_ext_, start, j);
						j ++;
                    }
                }
            }
		}

		void ThreadBackwardOp(int start, const int end) {
			/* Parameters : starting filter index | ending filter index
			 * Create offsets for slicing tensors
			 * Convolve backward to fill in delta
			 * Set appropriate values for X-offset and K-offset
			 * No mutex is needed as operations do not need to wait
			*/

			Eigen::DSizes<ptrdiff_t, 3> input_off {0, 0, 0};
			Eigen::DSizes<ptrdiff_t, 4> kernel_off {0, 0, 0, 0};
            int j = start * rows_ * cols_;

	        for (start; start < end; start ++) {
                kernel_off[0] = start;
                for (int r = 0; r < rows_; r ++) {
                    input_off[1] = r * height_stride_;
                    for (int c = 0; c < cols_; c ++) {
                        input_off[2] = c * width_stride_;
						//gradient for kernel
                        dweight_(start) = dweight_(start) + ((Eigen::Tensor<float, 0>)(input_.slice(input_off, input_ext_).sum()))(0) * flat_doutput_(j);
						//fill in layer gradient
                        layer_3d_delta_.slice(input_off, input_ext_) = layer_3d_delta_.slice(input_off, input_ext_) + kernel_.slice(kernel_off, kernel_ext_).reshape(input_ext_) * flat_doutput_(j);
                        j ++;
                    }
                }
				//optimization parameter for kernel gradient (Adam Props)
                vW_(start) = 0.1 * vW_(start) + 0.9 * (dweight_(start) * dweight_(start));
				//perform change to kernel
				kernel_.slice(kernel_off, kernel_ext_) = kernel_.slice(kernel_off, kernel_ext_) - lrate_ / sqrtf(vW_(start) + erate_) * dweight_(start);
            }
		}

};

}

#endif
