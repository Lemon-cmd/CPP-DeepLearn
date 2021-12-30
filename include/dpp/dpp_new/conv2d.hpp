#ifndef CONV2D_HPP
#define CONV2D_HPP

#include "layer.hpp"

namespace dpp {

class Conv2D : public Layer {

	/* 
	 * Convolutional Layer 2D
	 */

	public:
		Conv2D (const int num_krnls = 1, const int krnl_height = 1, const int krnl_width = 1, const int height_stride = 1, const int width_stride = 1,
                const std::string activation = "normal", const float lrate = 0.001, const float erate = 1e-8) {
			/* Save the parameters, and initalize parallelism and set the appropriate activation and recurrent activation functions */
            assert (num_krnls > 0 && krnl_height > 0 && krnl_width > 0 && height_stride > 0 && width_stride > 0 && lrate > 0.0 && erate > 0.0);
              
            // set learning and optimization rate
            lrate_ = lrate;
            erate_ = erate;
			out_dim_[0] = num_krnls;

            // set ext of krnl and input slice 
			in_ext_[1] = krnl_height; 
			in_ext_[2] = krnl_width;
			
			krnl_ext_[2] = krnl_height; 
			krnl_ext_[3] = krnl_width;
            
			// set stride and save krnl size to output
            stride_[0] = height_stride; 
			stride_[1] = width_stride; 
  
            // save the activation name for when saving to a file
            activation_ = activation == "normal" ? activation : activation == "tanh" ? activation : activation == "softmax" ? activation : activation == "relu" ? activation : "sigmoid";
  
            // make any necessary changes to activation function
			SetActivation(activate_, activation_);
        }

		const std::string type() { return "3D"; }

		const std::string name() { return "Conv2D"; }

		const Tensor3d& Get3dDelta() override { return delta_; }

		const Tensor3d& Get3dOutput() override { return output_; }

		const Dim3d Get3dOutputDim() override { return out_dim_; }

		void Topology() {
			std::cout << "\n\t-------- 2D-Convolutional Layer --------\n"
					  << "\n\tStride (Y, X):\t(" << stride_[0] << ", " << stride_[1] << ')'
					  << "\n\tParameters:\t(" << out_dim_[0] + out_dim_[0] * krnl_ext_[1] * krnl_ext_[2] * krnl_ext_[3] << ')'
					  << "\n\tInput-dim:\t(" << in_dim_[0] << ", " << in_dim_[1] << ", " << in_dim_[2] << ')'
					  << "\n\tOutput-dim:\t(" << out_dim_[0] << ", " << out_dim_[1] << ", " << out_dim_[2] << ")\n"
					  << "\n\t----------------------------------------\n\n";
        }

		void init(const Dim3d in_dim) override {
			/* Initialize the parameters of the layer and save the input shape */
            assert(in_dim[0] > 0 && in_dim[1] > 0 && in_dim[2] > 0 && in_dim[1] >= krnl_ext_[2] && in_dim[2] >= krnl_ext_[3]);
			// set initialized flag
			init_flag_ = true;

			// save input and calculate output shape
			in_dim_ = in_dim;
			out_dim_[1] = (in_dim_[1] - krnl_ext_[2]) / stride_[0] + 1;
			out_dim_[2] = (in_dim_[2] - krnl_ext_[3]) / stride_[1] + 1;

			// save channel size of input
			in_ext_[0] = in_dim_[0]; 
			krnl_ext_[1] = in_dim_[0];
			
			// flatten H and dH
			output1d_ = Tensor1d {out_dim_[0] * out_dim_[1] * out_dim_[2]};
			delta1d_ = Tensor1d {output1d_.dimension(0)};

			// set shape of layer gradient
			delta_ = Tensor3d {in_dim_};

			// momentum of weight and bias 
			vw_ = Tensor1d {out_dim_[0]}; vw_.setConstant(0.0);
			vb_ = Tensor1d {out_dim_[0]}; vb_.setConstant(0.0); 

			// set shape of bias, its gradient and weight gradient matrices
			bias_ = Tensor1d {out_dim_[0]};
			krnl_ = Tensor4d {out_dim_[0], in_dim_[0], krnl_ext_[2], krnl_ext_[3]};

            /* set values of weight and momentums */
            bias_.setConstant(1.0);
            krnl_.setConstant(pow(out_dim_[0] * in_dim_[0] * krnl_ext_[2] * krnl_ext_[3], -4.0));
  
            // select the appropriate size of thread pool and calculate the spread in thread executions
			thread_pool_.resize(out_dim_[0] <= std::thread::hardware_concurrency() ? out_dim_[0] : out_dim_[0] % std::thread::hardware_concurrency() + 1);
			slope_ = out_dim_[0] / thread_pool_.size();
		}

		void SetDelta(const Tensor3d &delta) {
			dh_ = dh_ * delta;
		}

		void Forward(const Tensor3d &X) override {
			assert(X.dimension(0) == in_dim_[0] && X.dimension(1) == in_dim_[1] && X.dimension(2) == in_dim_[2] && init_flag_);
			/* Perform Convolution Operation (Concurrently)
			 * Create thread pool and perform convolution concurrently
			 * Perform convolution if there exists remainders
			 * Else, convolve serially
			 * Map output and its gradients to H and dH
		     */

			input_ = X;
			
			start_ = 0; end_ = slope_;
			for (int t = 0; t < thread_pool_.size(); t ++) {
				thread_pool_[t] = std::thread(&Conv2D::ThreadConvol, this, start_, end_);
				start_ += slope_; end_ += slope_;
			}

			for (int t = 0; t < thread_pool_.size(); t ++) {
				thread_pool_[t].join();
			}

			if (out_dim_[0] % thread_pool_.size() > 0) {
				ThreadConvol(start_, out_dim_[0]);
			}

			// reshape dH and H
			dh_ = delta1d_.reshape(out_dim_);
			output_ = output1d_.reshape(out_dim_);
		}

		void Update() { 
		/* Reset delta
         * Update B
         * Reshape dH to 1D matrix
         * Update K and set new delta to pass
         */
			static Dim2d sum_dim {1, 2};
			static Tensor1d db, dw;
			
			db = Tensor1d {out_dim_[0]};
			dw = Tensor1d {out_dim_[0]};	
			
			dw.setConstant(0.0);															// reset weight gradient	
			delta_.setConstant(0.0);														// reset layer gradient
			delta1d_ = dh_.reshape(delta1d_.dimensions());									// flatten dj * dh

            db = dh_.sum(sum_dim);															// calculate sum of dH over axis 1 and 2; shape becomes {num_of_filters, 1}
            vb_ = 0.1 * vb_ + 0.9 * db.square();											// calculate bias momentum
			bias_ = bias_ - lrate_ / (vb_ + erate_).sqrt() * db;							// change bias
			
			start_ = 0; end_ = slope_;
            for (int t = 0; t < thread_pool_.size(); t ++) {
                thread_pool_[t] = std::thread(&Conv2D::ThreadUpdate, this, 
											  std::ref(db), std::ref(dw), start_, end_);
                start_ += slope_; end_ += slope_;
            }
  
            for (int t = 0; t < thread_pool_.size(); t ++) {
                thread_pool_[t].join();
            }
            
			if (out_dim_[0] % thread_pool_.size() > 0) {
				ThreadUpdate(db, dw, start_, out_dim_[0]);
            }
		}

		const float MeanSquaredError(const Tensor3d &Y, float &accuracy) override {
			dh_ = (output_ - Y) * dh_;							// dH = (H - Y) * dH
			// convert H and YH into vector for comparison
			Eigen::Map<const VectorXf> Y1d(Y.data(), Y.size()), 
									   H1d(output_.data(), output_.size());

			accuracy = H1d.isApprox(Y1d, 0.1) ? accuracy + 1 : accuracy;
			return ((Tensor0d) (0.5 * (output_ - Y).square().sum()))(0);
		}
  
		void Save(std::ofstream &file) {
			/* Write the parameters to a file */
			assert(init_flag_);
            // convert kernel and bias into vector
            Eigen::Map<VectorXf> krnl_vec (krnl_.data(), krnl_.size()), 
								 bias_vec (bias_.data(), bias_.size());
			file << "\nConv2D\n";
			file << "Activation: " << activation_ << '\n';
			file << "Learning_rate: (" << lrate_ << ")\n";
			file << "Momentum_rate: (" << erate_ << ")\n";
			file << "Height_stride: (" << stride_[0] << ")\n";
			file << "Width_stride: (" << stride_[1] << ")\n";
			file << "Input_shape: [" << in_dim_[0] << ',' << in_dim_[1] << ',' << in_dim_[2] << "]\n";
			file << "Output_shape: [" << out_dim_[0] << ',' << out_dim_[1] << ',' << out_dim_[2] << "]\n";
			file << "Kernel_shape: [" << out_dim_[0] << ',' << krnl_ext_[1] << ',' << krnl_ext_[2] << ',' << krnl_ext_[3] << "]\n";
			file << "Bias: ["; WriteVector(bias_vec, file);
			file << "Weight: ["; WriteVector(krnl_vec, file);
		}
		
		void Load(const Dim3d &in_dim, const Dim3d &out_dim, const VectorXfVec &weight) override {
			/* set the parameters to metadata inside of the layer */
			assert(weight.size() == 2);

			init_flag_ = true;

			// save input and output shapes
			in_dim_ = in_dim; out_dim_ = out_dim;

			// save channel size to exts
			in_ext_[0] = in_dim_[0]; 
			krnl_ext_[1] = in_dim_[0];

			// flatten H and dH
			output1d_ = Tensor1d {out_dim_[0] * out_dim_[1] * out_dim_[2]};
			delta1d_ = Tensor1d {output1d_.dimension(0)};

			// set layer gradient shape
			delta_ = Tensor3d {in_dim_};

			// momentum of weight and bias 
			vw_ = Tensor1d {out_dim_[0]}; vw_.setConstant(1.0);
			vb_ = Tensor1d {out_dim_[0]}; vb_.setConstant(1.0); 

			// map the vectors to their corresponding matrix
			bias_ = TensorMap <Eigen::Tensor<const float, 1>> (weight[0].data(), Dim1d {out_dim_[0]});
			krnl_ = TensorMap <Eigen::Tensor<const float, 4>> (weight[1].data(), Dim4d {out_dim_[0], in_dim_[0], krnl_ext_[2], krnl_ext_[3]});

			// set values of matrices
			vb_.setConstant(0.0); vw_.setConstant(0.0);
			
			// set the size of thread pool and calculate the spread in thread executions
			thread_pool_.resize(out_dim_[0] <= std::thread::hardware_concurrency() ? out_dim_[0] : out_dim_[0] % std::thread::hardware_concurrency() + 1);
			slope_ = out_dim_[0] / thread_pool_.size();
        }

	private:
		Dim2d stride_;
		Dim4d krnl_ext_ {1, 0, 0, 0};
		Dim3d in_ext_, in_dim_, out_dim_; 

		Tensor4d krnl_;
		Tensor3d input_, output_, delta_, dh_;
		Tensor1d vw_, vb_, bias_, output1d_, delta1d_;  
		
		std::function <void (Tensor1d &, Tensor1d &, 
							 const Tensor3d &, const Tensor4d &, const Tensor1d &,
						     const Dim3d &, const Dim3d &, const Dim4d &, const Dim4d &,
                             const int &, const int &)> activate_; 


		void ThreadConvol(int start, const int end) {
			/* Parameters : starting filter index | ending filter index
			* Create offsets for slicing tensors
			* Convolve forward and fill in O and dO which are later reshaped into H and dH
			* Set appropriate values for X-offset and K-offset
			* No mutex is needed as operations do not need to wait
			*/

			Dim3d in_off {0, 0, 0};
			Dim4d krnl_off {0, 0, 0, 0};
			int j = start * out_dim_[1] * out_dim_[2];

			for (start; start < end; start ++) {
				krnl_off[0] = start;
				
				for (int row = 0; row < out_dim_[1]; row ++) {
					in_off[1] = row * stride_[0];
					
					for (int col = 0; col < out_dim_[2]; col ++) {
						in_off[2] = col * stride_[1];
						activate_(output1d_, delta1d_, input_, krnl_, bias_, 
								  in_off, in_ext_, krnl_off, krnl_ext_, start, j);
						j ++;
					}
				}
			}
		}

		void ThreadUpdate(Tensor1d &db, Tensor1d &dw, int start, const int end) {
		/* Parameters : starting filter index | ending filter index
		 * Create offsets for slicing tensors
		 * Convolve backward to fill in delta
		 * Set appropriate values for X-offset and K-offset
		 * No mutex is needed as operations do not need to wait
		 */

			Dim3d in_off {0, 0, 0};
			Dim4d krnl_off {0, 0, 0, 0};
			int j = start * out_dim_[1] * out_dim_[2];

			for (start; start < end; start ++) {
				krnl_off[0] = start;
			
				for (int row = 0; row < out_dim_[1]; row ++) {
					in_off[1] = row * stride_[0];
				
					for (int col = 0; col < out_dim_[2]; col ++) {
						in_off[2] = col * stride_[1];
						//gradient for kernel
						dw(start) = dw(start) + ((Tensor0d) input_.slice(in_off, in_ext_).sum())(0) * delta1d_(j);
						//fill in layer gradient
						delta_.slice(in_off, in_ext_) = delta_.slice(in_off, in_ext_) + krnl_.slice(krnl_off, krnl_ext_).reshape(in_ext_) * delta1d_(j);
						j ++;
					}
				}
				//optimization parameter for kernel gradient (Adam Props)
				vw_(start) = 0.1 * vw_(start) + 0.9 * (dw(start) * dw(start));
				//perform change to kernel
				krnl_.slice(krnl_off, krnl_ext_) = krnl_.slice(krnl_off, krnl_ext_) - lrate_ / sqrtf(vw_(start) + erate_) * dw(start);
			}

		}
};

}
#endif
