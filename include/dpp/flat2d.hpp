#ifndef FLAT2D_HPP
#define FLAT2D_HPP

namespace dpp {

class Flat2D : public Layer {

	/*
	 *
	 * Flatten Layer 2D
	 *
	 */

	public:
		Flat2D() {} // empty constructor
		
		void init(const Eigen::DSizes<ptrdiff_t, 3> input_shape) override {
			assert(input_shape[0] > 0 && input_shape[1] > 0 && input_shape[2] > 0);
			/*
             *
             * Initialize the parameters of the layer and save the input shape
             *
             */

			// set initialized flag
			init_flag_ = true;

			// save input and calculate output shape
			input_3d_shape_ = input_shape;
			flat_output_size_ = input_shape[0] * input_shape[1] * input_shape[2];
			output_3d_shape_ = Eigen::DSizes<ptrdiff_t, 3> {1, 1, flat_output_size_};
		}

		void topology() {
            std::cout << "\n\t----------- 2D-Flatten Layer -----------\n"
					  << "\n\tParameters:\t(0)\n"
					  << "\n\tInput-dim:\t(" << input_3d_shape_[0] << ", " << input_3d_shape_[1] << ", " << input_3d_shape_[2] << ")\n"
                      << "\n\tOutput-dim:\t(" << output_3d_shape_[0] << ", " << output_3d_shape_[1] << ", " << output_3d_shape_[2] << ")\n"
					  << "\n\t----------------------------------------\n\n";
		}

		void forward(const Eigen::Tensor<float, 3> &X) override {
			assert(X.dimension(0) == input_3d_shape_[0] && X.dimension(1) == input_3d_shape_[1] && X.dimension(2) == input_3d_shape_[2] && init_flag_);

			// flatten input
			layer_3d_output_ = X.reshape(output_3d_shape_);
		}

		void update() { /* empty function */}

		void set_delta(const Eigen::Tensor<float, 3> &prev_layer_delta) override {
			// reshape back into the original shape
			layer_3d_delta_ = prev_layer_delta.reshape(input_3d_shape_);
		}

		const std::string name() { return "Flat2D"; }

		const std::string type() { return "3D"; }

		void save(std::ofstream &file) {
			assert(init_flag_);
			
			// write to file
			file << "\nFlat2D\n";
			file << "Input_shape: [" << input_3d_shape_[0] << ',' << input_3d_shape_[1] << ',' << input_3d_shape_[2] << "]\n";
            file << "Output_shape: [" << output_3d_shape_[0] << ',' << output_3d_shape_[1] << ',' << output_3d_shape_[2] << "]\n";	
		}

		void load(const Eigen::DSizes<ptrdiff_t, 3> &input_shape, const Eigen::DSizes<ptrdiff_t, 3> &output_shape) {

			// set the parameters to metadata inside of the layer
			init_flag_ = true;

			// save input and output shape
			input_3d_shape_ = input_shape;
			output_3d_shape_ = output_shape;

			// set flat output size
			flat_output_size_ = output_3d_shape_[2];
		}
};

}

#endif
