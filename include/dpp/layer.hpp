#ifndef LAYER_HPP
#define LAYER_HPP

namespace dpp {

class Layer {
	/* * 
	 *
	 * Abstract class Layer 
	 *
	 * */

	public:
		~Layer() { if (thread_pool_.size() > 0) {thread_pool_.clear();} }   // destructor; clear thread pool

		virtual void update() = 0;		 // update method for layer
		virtual void topology() = 0;	 // summary method lists parameters and input and output shapes

		virtual const std::string name() = 0;				// return name of the layer
		virtual const std::string type() = 0;				// return layer type (2D | 3D | 4D) to indicate appropriate return type for H and dH in sequential.hpp	
		virtual void save(std::ofstream &file) = 0;			// save layer method : write parameters to file

		virtual void reset() {}			// reset method for sequence layers (e.g., LSTM andd RNN); reset previous states

		virtual void forward(const Eigen::MatrixXf &X) {}				// forward method contains three versions for the appropriate input shape
		virtual void forward(const Eigen::Tensor<float, 3> &X) {}	    // for Conv2D
		virtual void forward(const Eigen::Tensor<float, 4> &X) {}		// for Conv3D

		virtual void set_delta(const Eigen::MatrixXf &prev_layer_delta) {}				// set gradient method contains three versions for the appropriate layer type 
		virtual void set_delta(const Eigen::Tensor<float, 3> &prev_layer_delta) {}	    // retreive gradient from previous layer and set new doutput
		virtual void set_delta(const Eigen::Tensor<float, 4> &prev_layer_delta) {}

		virtual void init(const Eigen::DSizes<ptrdiff_t, 2> input_shape) {}				// init method : initialize the parameters of the layer
		virtual void init(const Eigen::DSizes<ptrdiff_t, 3> input_shape) {}				// three versions for three appropriate input shapes
		virtual void init(const Eigen::DSizes<ptrdiff_t, 4> input_shape) {}

		virtual const Eigen::MatrixXf& Get2dLayerH() { return layer_2d_output_; }						// return layer output
		virtual const Eigen::Tensor<float, 3>& Get3dLayerH() { return layer_3d_output_; }
		virtual const Eigen::Tensor<float, 4>& Get4dLayerH() { return layer_4d_output_; }

		virtual const Eigen::MatrixXf& Get2dLayerDelta() { return layer_2d_delta_; }					// return layer gradient
		virtual const Eigen::Tensor<float, 3>& Get3dLayerDelta() { return layer_3d_delta_; }
		virtual const Eigen::Tensor<float, 4>& Get4dLayerDelta() { return layer_4d_delta_; }

		virtual const Eigen::DSizes<ptrdiff_t, 2> Get2dOutputShape() { return output_2d_shape_; }		// return output shape of layer 
		virtual const Eigen::DSizes<ptrdiff_t, 3> Get3dOutputShape() { return output_3d_shape_; }		// method is used for the init method of layers and compile method in sequential.hpp 
		virtual const Eigen::DSizes<ptrdiff_t, 4> Get4dOutputShape() { return output_4d_shape_; }

		/* Cost Functions : Categorical Cross Entropy and Mean Squared Error */
		virtual const float CrossEntropyError(const Eigen::MatrixXf &Y, float &accuracy) {}				
		virtual const float CrossEntropyError(const Eigen::Tensor<float, 3> &Y, float &accuracy) {}
		virtual const float CrossEntropyError(const Eigen::Tensor<float, 4> &Y, float &accuracy) {}

		virtual const float MeanSquaredError(const Eigen::MatrixXf &Y, float &accuracy) {}
		virtual const float MeanSquaredError(const Eigen::Tensor<float, 3> &Y, float &accuracy) {}
		virtual const float MeanSquaredError(const Eigen::Tensor<float, 4> &Y, float &accuracy) {}

		/* *
		 * 
		 * Load Method
		 * 
		 * There are multiple versions of this method due to the variety of layers
		 * */

		virtual void load(const Eigen::DSizes<ptrdiff_t, 2> &input_shape,
						  const Eigen::DSizes<ptrdiff_t, 2> &output_shape,
						  const std::vector<float> &bias_igate, const std::vector<float> &weight_igate, const std::vector<float> &u_weight_igate,
						  const std::vector<float> &bias_h, std::vector<float> &weight_h) {}

		virtual void load(const Eigen::DSizes<ptrdiff_t, 2> &input_shape,
						  const Eigen::DSizes<ptrdiff_t, 2> &output_shape,
						  const std::vector<float> &bias_fgate, const std::vector<float> &weight_fgate, const std::vector<float> &u_weight_fgate,
						  const std::vector<float> &bias_igate, const std::vector<float> &weight_igate, const std::vector<float> &u_weight_igate,
						  const std::vector<float> &bias_ogate, const std::vector<float> &weight_ogate, const std::vector<float> &u_weight_ogate,
						  const std::vector<float> &bias_cgate, const std::vector<float> &weight_cgate, const std::vector<float> &u_weight_cgate) {}

		virtual void load(const Eigen::DSizes<ptrdiff_t, 2> &input_shape, 
						  const Eigen::DSizes<ptrdiff_t, 2> &output_shape,  
						  const std::vector<float> &bias, 
						  const std::vector<float> &weight) {}

		virtual void load(const Eigen::DSizes<ptrdiff_t, 2> &input_shape, 
						  const Eigen::DSizes<ptrdiff_t, 2> &output_shape,
						  const std::vector<std::vector<float>> &weight) {}
		
		virtual void load(const Eigen::DSizes<ptrdiff_t, 3> &input_shape,
                          const Eigen::DSizes<ptrdiff_t, 3> &output_shape) {}

		virtual void load(const Eigen::DSizes<ptrdiff_t, 3> &input_shape,
						  const Eigen::DSizes<ptrdiff_t, 3> &output_shape,
						  std::vector<float> &bias,
						  std::vector<float> &weight) {}

	protected:
		bool init_flag_ = false;					// initialization flag of the layer
		
		float lrate_, erate_, mean_forward_;	    // learning rate, optimization lambda rate, mean of forward convolution (used in Convolutional)  
		
		int start_, end_, slope_,					// thread starting and ending index, and increment in their index 
			rows_, cols_, flat_output_size_;		// output shape height and width in convolution

		Eigen::MatrixXf layer_2d_output_, layer_2d_delta_;
		
		Eigen::Tensor<float, 3> layer_3d_output_, layer_3d_delta_, relu_3d_ones_, relu_3d_zeroes_;
		
		Eigen::Tensor<float, 4> layer_4d_output_, layer_4d_delta_, relu_4d_ones_, relu_4d_zeroes_;

		const Eigen::DSizes<ptrdiff_t, 2> sum_dim_ {1, 2};		// use in tensor summation

		Eigen::DSizes<ptrdiff_t, 2> input_2d_shape_, output_2d_shape_;
		
		Eigen::DSizes<ptrdiff_t, 3> input_3d_shape_, output_3d_shape_;
		
		Eigen::DSizes<ptrdiff_t, 4> input_4d_shape_, output_4d_shape_;

		std::mutex thread_lock_;
		
		std::vector<std::thread> thread_pool_;


		/* *
		 *
		 * Block Activation Functions 
		 * 
		 * Activate a specific row of the corresponding matrices
		 *
		 * */

		void block_relu(Eigen::MatrixXf &H, Eigen::MatrixXf &dH, const size_t &index) {
			H.row(index) = H.row(index).cwiseMax(0.0);
			dH.row(index) = H.row(index).cwiseMin(1.0);
		}

		void block_tanh(Eigen::MatrixXf &H, Eigen::MatrixXf &dH, const size_t &index) {
			H.row(index) = H.row(index).array().tanh();
			dH.row(index) = H.row(index).array() * (1.0 - H.row(index).array());
		}
		
		void block_normal(Eigen::MatrixXf &H, Eigen::MatrixXf &dH, const size_t &index) {
			dH.row(index) = H.row(index).array() / H.row(index).array();
		}

		void block_sigmoid(Eigen::MatrixXf &H, Eigen::MatrixXf &dH, const size_t &index) {
			H.row(index) = 1.0 / (1.0 + (-H).row(index).array().exp());
			dH.row(index) = H.row(index).array() * (1.0 - H.row(index).array());
		}
		
		void block_softmax(Eigen::MatrixXf &H, Eigen::MatrixXf &dH, const size_t &index) {
			H.row(index) = (H.row(index).array() - H.row(index).maxCoeff()).array().exp();
			H.row(index) = H.row(index) / H.row(index).sum();
			dH.row(index) = H.row(index).array() * (1.0 - H.row(index).array());
		}

		/* *
		 *
		 * Matrix Activation Functions 
		 * 
		 * Activate the entire matrix H and set its derivative to dH
		 * 		 
		 * */


		void matrix_relu(Eigen::MatrixXf &H, Eigen::MatrixXf &dH) {
			H = H.cwiseMax(0.0);
			dH = H.cwiseMin(1.0);
		}
		
		void matrix_tanh(Eigen::MatrixXf &H, Eigen::MatrixXf &dH) {
			H = H.array().tanh();
			dH = H.array() * (1.0 - H.array());
		}
		
		void matrix_normal(Eigen::MatrixXf &H, Eigen::MatrixXf &dH) {
			dH = Eigen::MatrixXf::Ones(H.rows(), H.cols());
		}

		void matrix_sigmoid(Eigen::MatrixXf &H, Eigen::MatrixXf &dH) {
			H = 1.0 / (1.0 + (-H).array().exp());
			dH = H.array() * (1.0 - H.array());
		}
		
		void matrix_softmax(Eigen::MatrixXf &H, Eigen::MatrixXf &dH) {
			H = (H.array() - H.maxCoeff()).array().exp();
			H = (H / H.sum()).eval();
			dH = H.array() * (1.0 - H.array());
		}

		/* *
		 *
		 * Tensor Activation Functions 
		 * 
		 * Activate the entire tensor H and set its derivative to dH
		 * 		 
		 * */

		void tensor_relu(Eigen::Tensor<float, 3> &H, Eigen::Tensor<float, 3> &dH) {
			H = H.cwiseMax(relu_3d_zeroes_);
			dH = H.cwiseMin(relu_3d_ones_);
		}

		void tensor_tanh(Eigen::Tensor<float, 3> &H, Eigen::Tensor<float, 3> &dH) {
            H = H.tanh();
            dH = H * (1.0 - H);
		} 

		void tensor_normal(Eigen::Tensor<float, 3> &H, Eigen::Tensor<float, 3> &dH) {
			dH = H.setConstant(1.0);
		}

		void tensor_sigmoid(Eigen::Tensor<float, 3> &H, Eigen::Tensor<float, 3> &dH) {
            H = 1.0 / (1.0 + (-H).exp());
            dH = H * (1.0 - H);
		} 
		
		void tensor_softmax(Eigen::Tensor<float, 3> &H, Eigen::Tensor<float, 3> &dH) {
            H = H.exp();
			H = H / ((Eigen::Tensor<float, 0>) H.sum())(0); 
			dH = H * (1.0 - H);
		} 
		
		/* *
		 *
		 * Convolutional Activation Functions (3D)
		 * 
		 * Activate a slice of H and set its derivative to a slice of dH
		 * Output and doutput are of the designated flat output size
		 * 		 
		 * */

		void conv_relu(const Eigen::Tensor<float, 3> &X,
					   const Eigen::Tensor<float, 4> &K,
					   const Eigen::Tensor<float, 1> &B,
					   Eigen::Tensor<float, 1> &O,
					   Eigen::Tensor<float, 1> &dO,
					   const Eigen::DSizes<ptrdiff_t, 3> &X_off, const Eigen::DSizes<ptrdiff_t, 3> &X_ext,
					   const Eigen::DSizes<ptrdiff_t, 4> &K_off, const Eigen::DSizes<ptrdiff_t, 4> &K_ext, 		 
					   const int &start, const int &j) {

			Eigen::Tensor<float, 3> product = (X.slice(X_off, X_ext) * K.slice(K_off, K_ext).reshape(X_ext)).cwiseMax(relu_3d_zeroes_);
			
			O(j) = ((Eigen::Tensor<float, 0>) product.sum())(0) * mean_forward_ + B(start);
			
			dO(j) = ((Eigen::Tensor<float, 0>) product.cwiseMin(relu_3d_ones_).sum())(0) * mean_forward_; 
		}

		void conv_tanh(const Eigen::Tensor<float, 3> &X,
                       const Eigen::Tensor<float, 4> &K,
                       const Eigen::Tensor<float, 1> &B,
                       Eigen::Tensor<float, 1> &O,
                       Eigen::Tensor<float, 1> &dO,
                       const Eigen::DSizes<ptrdiff_t, 3> &X_off, const Eigen::DSizes<ptrdiff_t, 3> &X_ext,
                       const Eigen::DSizes<ptrdiff_t, 4> &K_off, const Eigen::DSizes<ptrdiff_t, 4> &K_ext,
                       const int &start, const int &j) {

            Eigen::Tensor<float, 3> product = (X.slice(X_off, X_ext) * K.slice(K_off, K_ext).reshape(X_ext)).tanh();
            
			O(j) = ((Eigen::Tensor<float, 0>) product.sum())(0) * mean_forward_ + B(start);
            
			dO(j) = ((Eigen::Tensor<float, 0>) (product * (1.0 - product)).sum())(0) * mean_forward_;
        }

		void conv_normal(const Eigen::Tensor<float, 3> &X,
                         const Eigen::Tensor<float, 4> &K,                     
                         const Eigen::Tensor<float, 1> &B,                     
                         Eigen::Tensor<float, 1> &O,                     
                         Eigen::Tensor<float, 1> &dO,                    
                         const Eigen::DSizes<ptrdiff_t, 3> &X_off, const Eigen::DSizes<ptrdiff_t, 3> &X_ext,  
                         const Eigen::DSizes<ptrdiff_t, 4> &K_off, const Eigen::DSizes<ptrdiff_t, 4> &K_ext,  
                         const int &start, const int &j) {                     
                                                                                 
            O(j) = ((Eigen::Tensor<float, 0>)(X.slice(X_off, X_ext) * K.slice(K_off, K_ext).reshape(X_ext)).sum())(0) * mean_forward_ + B(start);
            
			dO(j) = mean_forward_;
        }  
		
		void conv_sigmoid(const Eigen::Tensor<float, 3> &X,
                          const Eigen::Tensor<float, 4> &K,
                          const Eigen::Tensor<float, 1> &B,
                          Eigen::Tensor<float, 1> &O,                     
                          Eigen::Tensor<float, 1> &dO,                    
                          const Eigen::DSizes<ptrdiff_t, 3> &X_off, const Eigen::DSizes<ptrdiff_t, 3> &X_ext,  
                          const Eigen::DSizes<ptrdiff_t, 4> &K_off, const Eigen::DSizes<ptrdiff_t, 4> &K_ext,  
                          const int &start, const int &j) {                     
                                                                                 
			Eigen::Tensor<float, 3> product = 1.0 / (1.0 + (-X.slice(X_off, X_ext) * K.slice(K_off, K_ext).reshape(X_ext)).exp());
            
			O(j) = ((Eigen::Tensor<float, 0>) product.sum())(0) * mean_forward_ + B(start);   
            
			dO(j) = ((Eigen::Tensor<float, 0>) (product * (1.0 - product)).sum())(0) * mean_forward_;
        }

		void conv_softmax(const Eigen::Tensor<float, 3> &X,
                          const Eigen::Tensor<float, 4> &K,
                          const Eigen::Tensor<float, 1> &B,
                          Eigen::Tensor<float, 1> &O,
                          Eigen::Tensor<float, 1> &dO,
                          const Eigen::DSizes<ptrdiff_t, 3> &X_off, const Eigen::DSizes<ptrdiff_t, 3> &X_ext,
                          const Eigen::DSizes<ptrdiff_t, 4> &K_off, const Eigen::DSizes<ptrdiff_t, 4> &K_ext,
                          const int &start, const int &j) {

            Eigen::Tensor<float, 3> product = (X.slice(X_off, X_ext) * K.slice(K_off, K_ext).reshape(X_ext)).exp();
			
			product = product / product.sum();
            
			O(j) = ((Eigen::Tensor<float, 0>) product.sum())(0) * mean_forward_ + B(start);
            
			dO(j) = ((Eigen::Tensor<float, 0>) (product * (1.0 - product)).sum())(0) * mean_forward_;
        }

		/* *
		 * 
		 *
		 * Arg Max Method 
		 * 
		 * Return the index of the maximum value within the far left column of the corresponding matrix or tensor
		 * 
		 *
		 * */

		void argmax4d(const Eigen::Tensor<float, 4> &target, float &max_value, const size_t &index) {
			for (int start = 0; start < target.dimension(3); start ++) {
				if (target(0, 0, index, start) == max_value) { max_value = start; break; }
			}
		}

		void argmax3d(const Eigen::Tensor<float, 3> &target, float &max_value, const size_t &index) {
			for (int start = 0; start < target.dimension(2); start ++) {
				if (target(0, index, start) == max_value) { max_value = start; break; }
			}
		}

		void argmax2d(const Eigen::MatrixXf &target, float &max_value, const size_t &index) {
			for (int start = 0; start < target.cols(); start ++) {
				if (target(index, start) == max_value) { max_value = start; break; }
			}
		}

		/* *
		 *
		 *
		 * Measure Methods
		 * Launch two threads asynchronously calling argmax method for comparing H and Y
		 * Note: This method is utilized within a thread pool inside of the CrossEntropyMethod
		 *
		 *
		 * */
		
		void measure4d(const Eigen::Tensor<float, 4> &H, const Eigen::Tensor<float, 4> &Y, float &hits, size_t index, const size_t rows) {
			float prediction, actual;
			Eigen::DSizes<ptrdiff_t, 4> offset {0, 0, 0, 0}, ext {1, 1, 1, H.dimension(3)};

			for (index; index < rows; index ++) {
				offset[2] = index;
				
				actual = ((Eigen::Tensor<float, 0>) Y.slice(offset, ext).maximum())(0);
				prediction = ((Eigen::Tensor<float, 0>) H.slice(offset, ext).maximum())(0);

				auto f1 = std::async(std::launch::deferred, &Layer::argmax4d, this, std::ref(Y), std::ref(actual), std::ref(index));
				auto f2 = std::async(std::launch::deferred, &Layer::argmax4d, this, std::ref(H), std::ref(prediction), std::ref(index));
				
				f1.get(); f2.get();

				thread_lock_.lock();
				hits = prediction == actual ? hits + 1 : hits;
				thread_lock_.unlock();
			}
		}

		void measure3d(const Eigen::Tensor<float, 3> &H, const Eigen::Tensor<float, 3> &Y, float &hits, size_t index, const size_t rows) {
			float prediction, actual;
			Eigen::DSizes<ptrdiff_t, 3> offset {0, 0, 0}, ext {1, 1, H.dimension(2)};

			for (index; index < rows; index ++) {
				offset[1] = index;
				
				actual = ((Eigen::Tensor<float, 0>) Y.slice(offset, ext).maximum())(0);
				prediction = ((Eigen::Tensor<float, 0>) H.slice(offset, ext).maximum())(0);	
				
				auto f1 = std::async(std::launch::deferred, &Layer::argmax3d, this, std::ref(Y), std::ref(actual), std::ref(index));
				auto f2 = std::async(std::launch::deferred, &Layer::argmax3d, this, std::ref(H), std::ref(prediction), std::ref(index));
				
				f1.get(); f2.get();
				
				thread_lock_.lock();
				hits = prediction == actual ? hits + 1 : hits;
				thread_lock_.unlock();
			}
		}

		void measure2d(const Eigen::MatrixXf &H, const Eigen::MatrixXf &Y, float &hits, size_t index, const size_t rows) {
			float prediction, actual;

			for (index; index < rows; index ++) {
				actual = Y.row(index).maxCoeff();
				prediction = H.row(index).maxCoeff();
				
				auto f1 = std::async(std::launch::deferred, &Layer::argmax2d, this, std::ref(H), std::ref(prediction), std::ref(index));
				auto f2 = std::async(std::launch::deferred, &Layer::argmax2d, this, std::ref(Y), std::ref(actual), std::ref(index));
				
				f1.get(); f2.get();
				
				thread_lock_.lock();
				hits = prediction == actual ? hits + 1 : hits;
				thread_lock_.unlock();
			}
		}
		
		/* *
		 *
		 * Print Shape Method 
		 *
		 * Output the shape of a matrix or a tensor
		 *
		 * 
		 * */

		void print_shape(const Eigen::MatrixXf &matrix) { std::cout << "Shape:\t(" << matrix.rows() << ", " << matrix.cols() << ")\n"; }

		void print_shape(const Eigen::Tensor<float, 3> &tensor3d) { 
			std::cout << "Shape:\t(" << tensor3d.dimension(0) << ", " 
									 << tensor3d.dimension(1) << ", " 
								     << tensor3d.dimension(2) << ")\n"; 
		}

		void print_shape(const Eigen::Tensor<float, 4> &tensor4d) { 
			std::cout << "Shape:\t(" << tensor4d.dimension(0) << ", " 
								     << tensor4d.dimension(1) << ", " 
									 << tensor4d.dimension(2) << ", " 
									 << tensor4d.dimension(3) << ")\n"; 
		}
		
		/* *
		 *
		 * Write Vector to File Method
		 *
		 * */

		void write_vector(const Eigen::VectorXf &target, std::ostream &file) {
			for (int j = 0; j < target.size() - 1; j ++) {
				file << target[j] << ',';
			}

			file << target[target.size() - 1] << "]\n";
		}
};
}

#endif
