#ifndef LAYER_HPP
#define LAYER_HPP

#include "modules.hpp"

namespace dpp {

using std::cout;
using Eigen::MatrixXf;
using Eigen::VectorXf;
using Eigen::TensorMap;
#define Tensor0d Eigen::Tensor<float, 0>
#define Tensor1d Eigen::Tensor<float, 1>
#define Tensor3d Eigen::Tensor<float, 3>
#define Tensor4d Eigen::Tensor<float, 4>
#define VectorXfVec std::vector<VectorXf>
#define MatrixXfVec	std::vector<MatrixXf>
#define Dim1d Eigen::DSizes<ptrdiff_t, 1>
#define Dim2d Eigen::DSizes<ptrdiff_t, 2>
#define Dim3d Eigen::DSizes<ptrdiff_t, 3>
#define Dim4d Eigen::DSizes<ptrdiff_t, 4>

class Layer {
	/* 
	 * Abstract Layer Class
	 */

	public:
		Layer() { Eigen::initParallel(); }
		~Layer() { if (thread_pool_.size() > 0) {thread_pool_.clear();} }				// destructor; clear thread pool

		virtual void Update() = 0;														// update method for layer
		virtual void Topology() = 0;													// summary method lists parameters and input and output shapes

		virtual const std::string name() = 0;											// return name of the layer
		virtual const std::string type() = 0;											// return layer type (2D | 3D | 4D) to indicate appropriate return type for H and dH in sequential.hpp

		virtual void Reset() {}															// reset method for sequence layers (e.g., LSTM andd RNN); reset previous states

		virtual void Save(std::ofstream &file) = 0;										// save layer method : write parameters to file

		virtual void Forward(const MatrixXf &X) {}										// forward method contains three versions for the appropriate input shape
		virtual void Forward(const Tensor3d &X) {}										// for Conv2D
		virtual void Forward(const Tensor4d &X) {}										// for Conv3D

		virtual void SetDelta(const MatrixXf &delta) {}									// set gradient method contains three versions for the appropriate layer type 
		virtual void SetDelta(const Tensor3d &delta) {}									// retreive gradient from previous layer and set new doutput
		virtual void SetDelta(const Tensor4d &delta) {}

		virtual void init(const Dim2d in_dim) {}										// init method : initialize the parameters of the layer
		virtual void init(const Dim3d in_dim) {}										// three versions for three appropriate input shapes
		virtual void init(const Dim4d in_dim) {}

		virtual const MatrixXf& Get2dOutput() {}										// return 2d-layer-output
		virtual const Tensor3d& Get3dOutput() {}										// return 3d-layer-output
		virtual const Tensor4d& Get4dOutput() {}										// return 4d-layer-output

		virtual const MatrixXf& Get2dDelta() {}											// return 2d-layer-gradient
		virtual const Tensor3d& Get3dDelta() {}											// return 3d-layer-gradient
		virtual const Tensor4d& Get4dDelta() {}											// return 4d-layer-gradient

		virtual const Dim2d Get2dOutputDim() {}											// return output shape of layer 
		virtual const Dim3d Get3dOutputDim() {}											// method is used for the init method of layers and compile method in sequential.hpp 
		virtual const Dim4d Get4dOutputDim() {}

		/* Cost Functions : Categorical Cross Entropy and Mean Squared Error */
		virtual const float CrossEntropyError(const MatrixXf &Y, float &accuracy) {}				
		virtual const float CrossEntropyError(const Tensor3d &Y, float &accuracy) {}
		virtual const float CrossEntropyError(const Tensor4d &Y, float &accuracy) {}

		virtual const float MeanSquaredError(const MatrixXf &Y, float &accuracy) {}
		virtual const float MeanSquaredError(const Tensor3d &Y, float &accuracy) {}
		virtual const float MeanSquaredError(const Tensor4d &Y, float &accuracy) {}
		
		/* *
		 * A variety of loading Methods
		 * */
		
		virtual void Load(const Dim2d &in_dim, const Dim2d &out_dim) {}
		virtual void Load(const Dim3d &in_dim, const Dim2d &out_dim) {}
		virtual void Load(const Dim4d &in_dim, const Dim2d &out_dim) {}
		virtual void Load(const Dim3d &in_dim, const Dim3d &out_dim) {}
		virtual void Load(const Dim2d &in_dim, const Dim2d &out_dim, const VectorXfVec &weight) {}
		virtual void Load(const Dim3d &in_dim, const Dim3d &out_dim, const VectorXfVec &weight) {}

	protected:
		float lrate_, erate_;										// learn rate and normalized momentum value
		bool init_flag_ = false;									// flag indicating if layer is initialized
		std::string activation_;									// gain function name

		std::mutex thread_lock_;									// mutex
		int start_, end_, slope_;									// thread's start-id, end-id, slope 
		std::vector<std::thread> thread_pool_;						// thread pool

		static float heaviside (float x) { return x > 0.0 ? 1.0 : 0.0; }

		/* *
		 *
		 * Sequence Activation Functions
		 * Activate a specific row of the corresponding matrices
		 *
		 * */

		void Relu1D(MatrixXf &H, MatrixXf &dH, const size_t &index) {
			H.row(index) = H.row(index).cwiseMax(0.0);
			dH.row(index) = H.row(index).unaryExpr(&heaviside);
		}

		void Tanh1D(MatrixXf &H, MatrixXf &dH, const size_t &index) {
			H.row(index) = H.row(index).array().tanh();
			dH.row(index) = 1.0 - H.row(index).array().square();
		}
		
		void Normal1D(MatrixXf &H, MatrixXf &dH, const size_t &index) {
			dH.row(index).setOnes();
		}

		void Sigmoid1D(MatrixXf &H, MatrixXf &dH, const size_t &index) {
			H.row(index) = 1.0 / (1.0 + (-H).row(index).array().exp());
			dH.row(index) = H.row(index).array() * (1.0 - H.row(index).array());
		}

		void Softsign1D(MatrixXf &H, MatrixXf &dH, const size_t &index) {
			dH.row(index) = H.row(index).array().abs();
			H.row(index) = H.row(index).array() / (1.0 + dH.row(index).array());
			dH.row(index) = 1.0 / (1.0 + dH.row(index).array()).array().square();
		}

		void Softmax1D(MatrixXf &H, MatrixXf &dH, const size_t &index) {
			H.row(index) = H.row(index).array().exp();
			H.row(index) = H.row(index) / H.row(index).sum();
			dH.row(index) = H.row(index).array() * (1.0 - H.row(index).array());
		}

		/* *
		 *
		 * Matrix Activation Functions
		 * Activate the entire matrix H and set its derivative to dH
		 *
		 * */

		void Relu2D(MatrixXf &H, MatrixXf &dH) {
			H = H.cwiseMax(0.0);
			dH = H.unaryExpr(&heaviside);
		}

		void Tanh2D(MatrixXf &H, MatrixXf &dH) {
			H = H.array().tanh();
			dH = 1.0 - H.array().square();
		}

		void Normal2D(MatrixXf &H, MatrixXf &dH) {
			dH = MatrixXf::Ones(H.rows(), H.cols());
		}

		void Sigmoid2D(MatrixXf &H, MatrixXf &dH) {
			H = 1.0 / (1.0 + (-H).array().exp());
			dH = H.array() * (1.0 - H.array());
		}

		void Softsign2D(MatrixXf &H, MatrixXf &dH) {
			dH = H.array().abs();
			H = H.array() / (1.0 + dH.array());
			dH = 1.0 / (1.0 + dH.array()).array().square();
		}

		void Softmax2D(MatrixXf &H, MatrixXf &dH) {
			H = H.array().exp();
			H = (H.array() / H.sum()).eval();
			dH = H.array() * (1.0 - H.array());
		}
		
		/* *
		 *
		 * 3D Tensor Activation Functions
		 * Activate the entire tensor H and set its derivative to dH
		 *
		 * */

		void Relu3D(Tensor3d &H, Tensor3d &dH) {
			H = H.cwiseMax(dH.setConstant(0.0));
			dH = H.unaryExpr(&heaviside);
		}

		void Tanh3D(Tensor3d &H, Tensor3d &dH) {
            H = H.tanh();
            dH = 1.0 - H.square();
		} 

		void Normal3D(Tensor3d &H, Tensor3d &dH) {
			dH = H.setConstant(1.0);
		}

		void Sigmoid3D(Tensor3d &H, Tensor3d &dH) {
            H = 1.0 / (1.0 + (-H).exp());
            dH = H * (1.0 - H);
		} 

		void Softsign3D(Tensor3d &H, Tensor3d &dH) {
			Tensor3d denom = 1.0 + H.abs(); 
			H = H / denom;	
			dH = 1.0 / denom.square();
		}
		
		void Softmax3D(Tensor3d &H, Tensor3d &dH) {
            H = H.exp();
			H = H / ((Tensor0d) H.sum())(0); 
			dH = H * (1.0 - H);
		}

	   /* *
	    *
	    * Convolutional Activation Functions (3D)
	    * Activate a slice of H and set its derivative to a slice of dH
	    * Output and doutput are of the designated flat output size
	    *
	    * */
		
		void ConvRelu3d(Tensor1d &H, 
						Tensor1d &dH,
						const Tensor3d &X, 
						const Tensor4d &K, 
						const Tensor1d &B,
					    const Dim3d &X_off, const Dim3d &X_ext,
					    const Dim4d &K_off, const Dim4d &K_ext,
					    const int &start, const int &j) {

			static Tensor3d result, zeros {X_ext}; 

			result = (X.slice(X_off, X_ext) * K.slice(K_off, K_ext).reshape(X_ext))
				     .cwiseMax(zeros);

			H(j) = ((Tensor0d) result.sum())(0) + B(start);

			dH(j) = ((Tensor0d) result.unaryExpr(&heaviside).sum())(0);
		}
		
		void ConvTanh3d(Tensor1d &H, 
						Tensor1d &dH,
						const Tensor3d &X, 
						const Tensor4d &K, 
						const Tensor1d &B,
					    const Dim3d &X_off, const Dim3d &X_ext,
					    const Dim4d &K_off, const Dim4d &K_ext,
					    const int &start, const int &j) {

			static Tensor3d result; 
			result = (X.slice(X_off, X_ext) * K.slice(K_off, K_ext).reshape(X_ext)).tanh();

			H(j) = ((Tensor0d) result.sum())(0) + B(start);

			dH(j) = ((Tensor0d) (1.0 - result.square()).sum())(0);
		}
		
		void ConvNormal3d(Tensor1d &H, 
						Tensor1d &dH,
						const Tensor3d &X, 
						const Tensor4d &K, 
						const Tensor1d &B,
					    const Dim3d &X_off, const Dim3d &X_ext,
					    const Dim4d &K_off, const Dim4d &K_ext,
					    const int &start, const int &j) {

			H(j) = ((Tensor0d) (X.slice(X_off, X_ext) * K.slice(K_off, K_ext).reshape(X_ext)).sum())(0);
			dH(j) = 1.0;
		}
		
		void ConvSigmoid3d(Tensor1d &H, 
						   Tensor1d &dH,
						   const Tensor3d &X, 
						   const Tensor4d &K, 
						   const Tensor1d &B,
					       const Dim3d &X_off, const Dim3d &X_ext,
					       const Dim4d &K_off, const Dim4d &K_ext,
					       const int &start, const int &j) {

			static Tensor3d result;	
			result = 1.0 / (1.0 + (-X.slice(X_off, X_ext) * K.slice(K_off, K_ext).reshape(X_ext)).exp());

			H(j) = ((Tensor0d) result.sum())(0) + B(start);
			
			dH(j) = ((Tensor0d) (result - result.square()).sum())(0);
		}
		
		void ConvSoftsign3d(Tensor1d &H, 
							Tensor1d &dH,
							const Tensor3d &X, 
							const Tensor4d &K, 
							const Tensor1d &B,
							const Dim3d &X_off, const Dim3d &X_ext,
							const Dim4d &K_off, const Dim4d &K_ext,
							const int &start, const int &j) {

			static Tensor3d result, denom; 
			result = X.slice(X_off, X_ext) * K.slice(K_off, K_ext).reshape(X_ext);

			denom = 1.0 + result.abs();

			H(j) = ((Tensor0d) (result / denom).sum())(0) + B(start);

			dH(j) = ((Tensor0d) (1.0 / denom.square()).sum())(0);
		}
		
		void ConvSoftmax3d(Tensor1d &H, 
						   Tensor1d &dH,
						   const Tensor3d &X, 
						   const Tensor4d &K, 
						   const Tensor1d &B,
					       const Dim3d &X_off, const Dim3d &X_ext,
					       const Dim4d &K_off, const Dim4d &K_ext,
					       const int &start, const int &j) {

			static Tensor3d result; 
			result = (X.slice(X_off, X_ext) * K.slice(K_off, K_ext).reshape(X_ext)).exp();
			result = result / result.sum();

			H(j) = ((Tensor0d) result.sum())(0) + B(start);

			dH(j) = ((Tensor0d) (1.0 - result.square()).sum())(0);
		}

		/*
		 *
		 * Set Activation Function Method
		 * Given an empty function and activation name
		 * Bind the appropriate activation function
		 *
		 */

		void SetActivation(std::function <void (MatrixXf &, MatrixXf &)> &act_fun, const std::string &activation) {
			act_fun = std::bind(&Layer::Normal2D, this, std::placeholders::_1, std::placeholders::_2);

			if (activation == "sigmoid") {
				act_fun = std::bind(&Layer::Sigmoid2D, this, std::placeholders::_1, std::placeholders::_2);
			} else if (activation == "relu") {
				act_fun = std::bind(&Layer::Relu2D, this, std::placeholders::_1, std::placeholders::_2);
			} else if (activation == "tanh") {
				act_fun = std::bind(&Layer::Tanh2D, this, std::placeholders::_1, std::placeholders::_2);
			} else if (activation == "softmax"){
				act_fun = std::bind(&Layer::Softmax2D, this, std::placeholders::_1, std::placeholders::_2);
			} else if (activation == "softsign") {
				act_fun = std::bind(&Layer::Softsign2D, this, std::placeholders::_1, std::placeholders::_2);
			}
		}

		void SetActivation(std::function <void (MatrixXf &, MatrixXf &, const int &)> &act_fun, const std::string &activation) {
			act_fun = std::bind(&Layer::Normal1D, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);

			if (activation == "sigmoid") {
				act_fun = std::bind(&Layer::Sigmoid1D, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
			} else if (activation == "relu") {
                act_fun = std::bind(&Layer::Relu1D, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
            } else if (activation == "tanh") {
                act_fun = std::bind(&Layer::Tanh1D, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
            } else if (activation == "softmax") {
                act_fun = std::bind(&Layer::Softmax1D, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
            } else if (activation == "softsign") {
				act_fun = std::bind(&Layer::Softsign1D, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
			}
		}

		void SetActivation(std::function <void (Tensor3d &, Tensor3d &)> &act_fun, const std::string &activation) {
			act_fun = std::bind(&Layer::Normal3D, this, std::placeholders::_1, std::placeholders::_2);

			if (activation == "sigmoid") {
				act_fun = std::bind(&Layer::Sigmoid3D, this, std::placeholders::_1, std::placeholders::_2);
			} else if (activation == "relu") {
				act_fun = std::bind(&Layer::Relu3D, this, std::placeholders::_1, std::placeholders::_2);
			} else if (activation == "tanh") {
				act_fun = std::bind(&Layer::Tanh3D, this, std::placeholders::_1, std::placeholders::_2);
			} else if (activation == "softmax") {
				act_fun = std::bind(&Layer::Softmax3D, this, std::placeholders::_1, std::placeholders::_2);
			} else if (activation == "softsign") {
				act_fun = std::bind(&Layer::Softsign3D, this, std::placeholders::_1, std::placeholders::_2);
			}
		}

		void SetActivation(std::function <void (Tensor1d &, Tensor1d &,
						   const Tensor3d &, const Tensor4d &, const Tensor1d &,
						   const Dim3d &, const Dim3d &, const Dim4d &, const Dim4d &,
                           const int &, const int &)> &act_fun, const std::string &activation) {

			act_fun = std::bind(&Layer::ConvNormal3d, this, std::placeholders::_1,
								std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5, std::placeholders::_6,
								std::placeholders::_7, std::placeholders::_8, std::placeholders::_9, std::placeholders::_10, std::placeholders::_11);

			if (activation == "relu") {
				act_fun = std::bind(&Layer::ConvRelu3d, this, std::placeholders::_1,
									  std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5, std::placeholders::_6,
                                      std::placeholders::_7, std::placeholders::_8, std::placeholders::_9, std::placeholders::_10, std::placeholders::_11);
			} else if (activation == "tanh") {
				act_fun = std::bind(&Layer::ConvTanh3d, this, std::placeholders::_1,
                                      std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5, std::placeholders::_6,
                                      std::placeholders::_7, std::placeholders::_8, std::placeholders::_9, std::placeholders::_10, std::placeholders::_11);
			} else if (activation == "sigmoid") {
			    act_fun = std::bind(&Layer::ConvSigmoid3d, this, std::placeholders::_1,
                                      std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5, std::placeholders::_6,
									  std::placeholders::_7, std::placeholders::_8, std::placeholders::_9, std::placeholders::_10, std::placeholders::_11);
			} else if (activation == "softmax") {
				act_fun = std::bind(&Layer::ConvSoftmax3d, this, std::placeholders::_1,
                                      std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5, std::placeholders::_6,
                                      std::placeholders::_7, std::placeholders::_8, std::placeholders::_9, std::placeholders::_10, std::placeholders::_11);
			} else if (activation == "softsign") {
				act_fun = std::bind(&Layer::ConvSoftsign3d, this, std::placeholders::_1,
                                      std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5, std::placeholders::_6,
                                      std::placeholders::_7, std::placeholders::_8, std::placeholders::_9, std::placeholders::_10, std::placeholders::_11);
			}
		}

		/* *
		 *
		 * Arg Max Method
		 * Return the index of the maximum value within the far left column of the corresponding matrix or tensor
		 *
		 * */

		void Argmax(const MatrixXf &target, float &max_value, const size_t &index) {
			for (int start = 0; start < target.cols(); start ++) {
				if (target(index, start) == max_value) { max_value = start; break; }
			}
		}
		
		/* *
		 *
		 * Measure Method
		 * Launch two threads asynchronously calling argmax method for comparing H and Y
		 * Note: This method is utilized within a thread pool inside of the CrossEntropyMethod
		 *
		 * */

		void Measure(const MatrixXf &H, const MatrixXf &Y, float &hits, size_t index, const size_t rows) {
			float prediction, actual;

			for (index; index < rows; index ++) {
				actual = Y.row(index).maxCoeff();
				prediction = H.row(index).maxCoeff();

				auto f1 = std::async(std::launch::async | std::launch::deferred, &Layer::Argmax, this, std::ref(H), std::ref(prediction), std::ref(index));
				auto f2 = std::async(std::launch::async | std::launch::deferred, &Layer::Argmax, this, std::ref(Y), std::ref(actual), std::ref(index));

				f1.get(); f2.get();

				thread_lock_.lock();
				hits = prediction == actual ? hits + 1 : hits;
				thread_lock_.unlock();
			}
		}
		
		/* *
		 * Print Shape Method 
		 * Output the shape of a matrix or a tensor
		 * */

		void GetShape(const Dim1d &dim) {
			cout << "Shape:\t(" << dim[0] << ")\n"; 
		}

		void GetShape(const Dim2d &dim) {
			cout << "Shape:\t(" << dim[0] << ", " 
									 << dim[1] << ", " << ")\n"; 
		}
		
		void GetShape(const Dim3d &dim) {
			cout << "Shape:\t(" << dim[0] << ", "
									 << dim[1] << ", " 
									 << dim[2] << ", " << ")\n"; 
		}
		
		void GetShape(const Dim4d &dim) {
			cout << "Shape:\t(" << dim[0] << ", " 
									 << dim[1] << ", " 
									 << dim[2] << ", " 
									 << dim[3] << ", " << ")\n"; 
		}

		void GetShape(const MatrixXf &matrix) { 
			cout << "Shape:\t(" << matrix.rows() << ", " << matrix.cols() << ")\n"; 
		}
		
		/*
		void GetShape(const Tensor1d &tensor) { 
			cout << "Shape:\t(" << tensor.dimension(0) << ")\n"; 
		}

		void GetShape(const Tensor2d &tensor) { 
			cout << "Shape:\t(" << tensor.dimension(0) << ", " 
								     << tensor.dimension(1) << ")\n"; 
		}
		*/

		void GetShape(const Tensor3d &tensor) { 
			cout << "Shape:\t(" << tensor.dimension(0) << ", " 
									 << tensor.dimension(1) << ", " 
								     << tensor.dimension(2) << ")\n"; 
		}

		void GetShape(const Tensor4d &tensor) { 
			cout << "Shape:\t(" << tensor.dimension(0) << ", " 
								     << tensor.dimension(1) << ", " 
									 << tensor.dimension(2) << ", " 
									 << tensor.dimension(3) << ")\n"; 
		}
		
		/* *
		 * Write Vector to File Method
		 * */

		void WriteVector(const VectorXf &target, std::ostream &file) {
			for (int j = 0; j < target.size() - 1; j ++) {
				file << target[j] << ',';
			}

			file << target[target.size() - 1] << "]\n";
		}

		template <typename T, typename std::enable_if <std::is_arithmetic<T>::value>::type* = nullptr>
		const T randint (const int min, const int max) {
			std::random_device rd;
			std::mt19937 gen(rd());
			std::uniform_int_distribution <T> dist(min, max - 1);
			return dist(gen);
		}

		void ResetMatVec(std::vector <MatrixXf> &v) {
			// loop through vector and reset matrix
			std::for_each(v.begin(), v.end(), [](MatrixXf &matrix) { matrix.setZero(); });
		}

};
}
#endif
