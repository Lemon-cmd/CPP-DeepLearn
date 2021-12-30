#ifndef FLATTEN_HPP
#define FLATTEN_HPP

#include "layer.hpp"

namespace dpp {

class Flatten : public Layer {
	/*
    *
    * Flatten Layer
    *
    */

    public:
        Flatten() {} // empty constructor

		void Update() {}

		const std::string type() { return "1D"; }
              
        const std::string name() { return "Flatten"; }

		const MatrixXf& Get2dDelta() override { return delta2d_; }
		
		const Tensor3d& Get3dDelta() override { return delta3d_; }

		const Tensor4d& Get4dDelta() override { return delta4d_; }
		
		const MatrixXf& Get2dOutput() override { return output_; }	
	
		const Dim2d Get2dOutputDim() override { return out_dim_; }
	
		void init(const Dim2d in_dim) override {
              assert(in_dim[0] > 0 && in_dim[1] > 0);
              /*
               * Initialize the parameters of the layer and save the input shape
               */

              // set initialized flags
              init_flag_ = true; flag2d_ = true;

              // save input and calculate output shape
              in_dim2d_ = in_dim;
  			  out_dim_= Dim2d {1, in_dim[0] * in_dim[1] * in_dim[2]};

			  // set other flags to false
			  flag3d_ = false; flag4d_ = false;
          }

		void init(const Dim3d in_dim) override {
              assert(in_dim[0] > 0 && in_dim[1] > 0 && in_dim[2] > 0);
              /*
               * Initialize the parameters of the layer and save the input shape
               */

              // set initialized flag
              init_flag_ = true; flag3d_ = true;

              // save input and calculate output shape
              in_dim3d_ = in_dim;
  			  out_dim_ = Dim2d {1, in_dim[0] * in_dim[1] * in_dim[2]};
			
			  // set other flags to false
			  flag2d_ = false; flag4d_ = false;
          }
		
		void init(const Dim4d in_dim) override {
              assert(in_dim[0] > 0 && in_dim[1] > 0 && in_dim[2] > 0 && in_dim[3] > 0);
              /*
               * Initialize the parameters of the layer and save the input shape
               */

              // set initialized flag
              init_flag_ = true; flag4d_ = true;

              // save input and calculate output shape
              in_dim4d_ = in_dim;
  			  out_dim_ = Dim2d {1, in_dim[0] * in_dim[1] * in_dim[2] * in_dim[3]};
				
			  // set other flags to false
			  flag2d_ = false; flag3d_ = false;
          }

		void Topology() {
			std::cout << "\n\t----------- 2D-Flatten Layer -----------\n"
					  << "\n\tParameters:\t(0)\n";
			if (flag3d_) {
				std::cout << "\n\tInput-dim:\t(" << in_dim3d_[0] << ", " << in_dim3d_[1] << ", " << in_dim3d_[2] << ")\n";
			} else if (flag4d_) {
				std::cout << "\n\tInput-dim:\t(" << in_dim4d_[0] << ", " << in_dim4d_[1] << ", " << in_dim4d_[2] << ", " << in_dim4d_[3] << ")\n";
			} else {
				std::cout << "\n\tInput-dim:\t(" << in_dim2d_[0] << ", " << in_dim2d_[1] << ")\n";
			}

			std::cout << "\n\tOutput-dim:\t(" << out_dim_[0] << ", " << out_dim_[1] << ")\n"
                      << "\n\t----------------------------------------\n\n";
		}

		void Forward(const MatrixXf &X) override {
			assert(X.rows() == in_dim2d_[0] && X.cols() == in_dim2d_[1] && init_flag_);	
			// flatten input
			output_ = Eigen::Map<const MatrixXf> (X.data(), out_dim_[0], out_dim_[1]);
		}

		void Forward(const Tensor3d &X) override {
			assert(X.dimension(0) == in_dim3d_[0] && X.dimension(1) == in_dim3d_[1] && X.dimension(2) == in_dim3d_[2] && init_flag_);
			// flatten input
            output_ = Eigen::Map<const MatrixXf> (X.data(), out_dim_[0], out_dim_[1]) ;
		}

		void Forward(const Tensor4d &X) override {
			assert(X.dimension(0) == in_dim4d_[0] && X.dimension(1) == in_dim4d_[1] && X.dimension(2) == in_dim4d_[2] && X.dimension(3) == in_dim4d_[3] && init_flag_);
			// flatten input
            output_ = Eigen::Map<const MatrixXf> (X.data(), out_dim_[0], out_dim_[1]) ;
		}

		void SetDelta(const MatrixXf &delta) override {
			if (flag3d_) {
				delta3d_ = Eigen::TensorMap<Eigen::Tensor<const float, 3>> (delta.data(), in_dim3d_);	
			} else if(flag4d_) {
				delta4d_ = Eigen::TensorMap<Eigen::Tensor<const float, 4>> (delta.data(), in_dim4d_);
			} else {
				delta2d_ = Eigen::Map<const MatrixXf> (delta.data(), in_dim2d_[0], in_dim2d_[1]);
			}
		}	
		
		void Save(std::ofstream &file) {
            assert(init_flag_);
            // write to file
            file << "\nFlatten Layer\n";

			if (flag2d_) {
				file << "Input_shape: [" << in_dim2d_[0] << ',' << in_dim2d_[1] << ',' << in_dim2d_[2] << "]\n";
				file << "Output_shape: [" << out_dim_[0] << ',' << out_dim_[1] << "]\n";
			} else if (flag3d_) {
				file << "Input_shape: [" << in_dim3d_[0] << ',' << in_dim3d_[1] << ',' << in_dim3d_[2] << "]\n";
				file << "Output_shape: [" << out_dim_[0] << ',' << out_dim_[1] << "]\n";
			} else {
				file << "Input_shape: [" << in_dim4d_[0] << ',' << in_dim4d_[1] << ',' << in_dim4d_[2] << "]\n";
				file << "Output_shape: [" << out_dim_[0] << ',' << out_dim_[1] << "]\n";
			}
        }

		void Load(const Dim2d &in_dim, const Dim2d &out_dim) {
            // set the parameters to metadata inside of the layer
            init_flag_ = true;

            // save input and output shape
            in_dim2d_ = in_dim;
            out_dim_ = out_dim;
        }

		void Load(const Dim3d &in_dim, const Dim2d &out_dim) {
            // set the parameters to metadata inside of the layer
            init_flag_ = true;

            // save input and output shape
            in_dim3d_ = in_dim;
            out_dim_ = out_dim;
        }
		
		void Load(const Dim4d &in_dim, const Dim2d &out_dim) {
            // set the parameters to metadata inside of the layer
            init_flag_ = true;

            // save input and output shape
            in_dim4d_ = in_dim;
            out_dim_ = out_dim;
        }

	private:
		Dim4d in_dim4d_;
		Dim3d in_dim3d_;	
		Dim2d in_dim2d_, out_dim_;
		bool flag2d_, flag3d_, flag4d_;
		
		Tensor3d delta3d_;
		Tensor4d delta4d_;
		MatrixXf delta2d_, output_;
};
}
#endif
