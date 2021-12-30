#ifndef POOL2D_HPP
#define POOL2D_HPP

#include "layer.hpp"

namespace dpp {

class Pool2D : public Layer {
	
	/* Pooling 2D Layer */ 

	public:
		Pool2D(const int krnl_height = 1, const int krnl_width = 1, const int height_stride = 1, const int width_stride = 1) {
			assert (krnl_height > 0 && krnl_width > 0 && height_stride > 0 && width_stride > 0);
			/* Save the parameters, initalize parallelism, and set the appropriate activation function */
			
			// save krnl width and height to ext for slicing
			in_ext_[1] = krnl_height; in_ext_[2] = krnl_width;

			// save width and height stride
			stride_[0] = height_stride; stride_[1] = width_stride;
		}
			
		const std::string name() { return "Pool2D"; }

		const std::string type() { return "3D"; } 

		const Tensor3d& Get3dDelta() override { return delta_; }

		const Tensor3d& Get3dOutput() override { return output_; }

		const Dim3d Get3dOutputDim() override { return out_dim_; }

		void Topology() {
			std::cout << "\n\t----------- 2D-Pooling Layer -----------\n"
					  << "\n\tStride (Y, X):\t(" << stride_[0] << ", " << stride_[1] << ')'
                      << "\n\tParameters:\t(" << 0 << ')'
					  << "\n\tInput-dim:\t(" << in_dim_[0] << ", " << in_dim_[1] << ", " << in_dim_[2] << ")\n"
                      << "\n\tOutput-dim:\t(" << out_dim_[0] << ", " << out_dim_[1] << ", " << out_dim_[2] << ")\n"
                      << "\n\t----------------------------------------\n\n";
		}
		
		void init(const Dim3d in_dim) override {
			/* Initialize the parameters of the layer */
			assert (in_dim[0] > 0 && in_dim[1] > 0 && in_dim[2] > 0 && in_dim[1] > in_ext_[1] && in_dim[2] > in_ext_[2]);
			// set initialized flag
			init_flag_ = true;	

			// save input
			in_dim_ = in_dim;
			
			// save channel size
			in_ext_[0] = in_dim_[0]; out_ext_[0] = in_dim_[0];

			// calculate output shape
			out_dim_[0] = in_dim_[0];
			out_dim_[1] = (in_dim_[1] - in_ext_[1]) / stride_[0] + 1;				// height
			out_dim_[2] = (in_dim_[2] - in_ext_[2]) / stride_[1] + 1;				// width
			
			// set flatten dH and H shape
			output1d_ = Tensor1d {out_dim_[0] * out_dim_[1] * out_dim_[2]};
			delta1d_ = Tensor1d {output1d_.dimension(0)};

			// set shape of H and layer delta
			delta_ = Tensor3d {in_dim_};
			
			// calculate mean of summation in pooling; this is used in calculating layer delta
            mean_ = 1.0 / (in_ext_[2] * in_ext_[1]);
			
			// select the appropriate size of thread pool and calculate the spread in thread executions
			thread_pool_.resize(out_dim_[1] <= std::thread::hardware_concurrency() ? out_dim_[1] : out_dim_[1] % std::thread::hardware_concurrency() + 1);
			slope_ = out_dim_[1] / thread_pool_.size();
		}

		void Forward(const Tensor3d &X) override {
			assert(X.dimension(0) == in_dim_[0] && X.dimension(1) == in_dim_[1] && X.dimension(2) == in_dim_[2] && init_flag_);
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
                thread_pool_[t] = std::thread(&Pool2D::ThreadForward, this, start_, end_);
                start_ += slope_; end_ += slope_;
            }
  
            for (int t = 0; t < thread_pool_.size(); t ++) {
                thread_pool_[t].join();
            }
  
            if (out_dim_[1] % thread_pool_.size() > 0) {
                ThreadForward(start_, out_dim_[1]);
            }
			
			// reshape flat H and dH to their intended size
			dh_ = delta1d_.reshape(out_dim_); 
			output_ = output1d_.reshape(out_dim_);
		}
		
		void SetDelta(const Tensor3d &delta) override {
            dh_ = delta * dh_;										// dH = gradient(L + 1) * dH(L)
        }

		void Update() {
			/*
            * Reset delta
            * Flatten dH
            * Convolve and fill delta
			*/

			// reset delta
			delta_.setConstant(0.0);

			// flatten new dH
			delta1d_ = dh_.reshape(delta1d_.dimensions());

			// reset index range of thread
			start_ = 0; end_ = slope_;
			for (int t = 0; t < thread_pool_.size(); t ++) {
                thread_pool_[t] = std::thread(&Pool2D::ThreadUpdate, this, start_, end_);

                start_ += slope_; end_ += slope_;
            }
    
            for (int t = 0; t < thread_pool_.size(); t ++) {
				thread_pool_[t].join();
            }
    
            if (out_dim_[1] % thread_pool_.size() > 0) {
                ThreadUpdate(start_, out_dim_[1]);
            } 
		}

        const float MeanSquaredError(const Tensor3d &Y, float &accuracy) override {
			// dH = (H - Y) * dH
            dh_ = (output_ - Y) * dh_;

			// convert Y and H into vector for comparison
			Eigen::Map<const VectorXf> Y1d(Y.data(), Y.size()), 
									   H1d(output_.data(), output_.size());

			accuracy = H1d.isApprox(Y1d, 0.1) ? accuracy + 1 : accuracy;
            return ((Tensor0d) (0.5 * (output_ - Y).square().sum()))(0);
        }

		void Save(std::ofstream &file) {
			assert(init_flag_);

			// write to file
			file << "\nPool2D\n";
			file << "Height_stride: (" << stride_[0] << ")\n";
			file << "Width_stride: (" << stride_[1] << ")\n";
			file << "Input_shape: [" << in_dim_[0] << ',' << in_dim_[1] << ',' << in_dim_[2] << "]\n";
            file << "Output_shape: [" << out_dim_[0] << ',' << out_dim_[1] << ',' << out_dim_[2] << "]\n";
			file << "Kernel_shape: [" << in_ext_[0] << ',' << in_ext_[1] << ',' << in_ext_[2] << "]\n";
		}

		void Load(const Dim3d &in_dim, const Dim3d &out_dim) {
			// set the parameters to metadata inside of the layer
			init_flag_ = true;

			// save input and output shapes
			in_dim_ = in_dim;
			out_dim_ = out_dim;

			// save channel size to ext for slicing
			in_ext_[0] = in_dim_[0]; out_ext_[0] = in_dim_[0];

			// calculate output shape
			out_dim_[0] = in_dim_[0];
			out_dim_[1] = (in_dim_[1] - in_ext_[1]) / stride_[0] + 1;
			out_dim_[2] = (in_dim_[2] - in_ext_[2]) / stride_[1] + 1;
			
			// set flatten dH and H shape
			output1d_ = Tensor1d {out_dim_[0] * out_dim_[1] * out_dim_[2]};
			delta1d_ = Tensor1d {output1d_.dimension(0)};
			
			// set layer delta shape
			delta_ = Tensor3d {in_dim_};
			
			// calculate mean of summation in pooling; this is used in calculating layer delta
            mean_ = 1.0 / (in_ext_[2] * in_ext_[1]);

			// set the size of thread pool and calculate the spread in thread executions
			thread_pool_.resize(out_dim_[1] <= std::thread::hardware_concurrency() ? out_dim_[1] : out_dim_[1] % std::thread::hardware_concurrency() + 1);
			slope_ = out_dim_[1] / thread_pool_.size();
		}	

	private:
		float mean_;

		Tensor1d output1d_, delta1d_;
		Tensor3d input_, output_, dh_, delta_;

		Dim1d out_ext_; 
		Dim2d stride_, sum_dim_ {1, 2};
		Dim3d in_ext_, in_dim_, out_dim_;

		void ThreadForward(int start, const int end) {
			/* Perform forward convolution
			 * Parameters : start index of rows | end index of rows
			 * Create offsets for output and input slicing
			 * Find maximum value of each channel in input
			 * Size of output's slice is equal to the number of channels
			*/

			int j = start * out_dim_[2];
			Dim1d out_off {0};
			Dim3d in_off {0, 0, 0};

			for (start; start < end; start ++) {
				in_off[1] = start * stride_[0];

				for (int col = 0; col < out_dim_[2]; col ++) {
					in_off[2] = col * stride_[1];
					
					out_off[0] = j * out_dim_[0];

					output1d_.slice(out_off, out_ext_) = input_.slice(in_off, in_ext_).maximum(sum_dim_);

					delta1d_.slice(out_off, out_ext_) = output1d_.slice(out_off, out_ext_) * mean_;
					
					j ++;
				}
			}
		}

		void ThreadUpdate(int start, const int end) {
			/*
			 * Perform backward convolution
			 * Parameters : start inddex of rows | end index of rows
			 * Create offset for input
			 * Fill in delta
			*/

			int j = start * out_dim_[2];
			Dim3d in_off {0, 0, 0};

			for (start; start < end; start ++) {
				in_off[1] = start * stride_[0];
				
				for (int col = 0; col < out_dim_[2]; col ++) {
					in_off[2] = col * stride_[1];
					
					delta_.slice(in_off, in_ext_) = delta_.slice(in_off, in_ext_) + delta1d_(j);

					j ++;
				}
			}
		}

};

}

#endif
