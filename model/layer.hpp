#include "modules.hpp"

class Layer {
	protected:
		Eigen::MatrixXf I, H, delta; std::string activate;
		const float epi = expf(-std::numbers::pi);
		void hrelu(Eigen::MatrixXf &Z, Eigen::MatrixXf &dZ) {
			const Eigen::MatrixXf epiZ = epi * Z.array();
			Z = epiZ.array().sinh() + epiZ.array().cosh();		
			dZ = epi * (epiZ.array().cosh() - epiZ.array().sinh()).array();
		}
		void sigmoid(Eigen::MatrixXf &Z, Eigen::MatrixXf &dZ) {
			Z = 1 / (1 + -Z.unaryExpr<float(*)(float)>(&expf).array()).array();
			dZ = Z.array() * (1 - Z.array()).array();
		}
		void softmax(Eigen::MatrixXf &Z, Eigen::MatrixXf &dZ) {
			const Eigen::MatrixXf top = (Z.array() - Z.maxCoeff()).unaryExpr<float(*)(float)>(&expf);
			Z = top.array() / top.sum();
			dZ = Z.array() * (1 - Z.array()).array();
		}
		void ztanh(Eigen::MatrixXf &Z, Eigen::MatrixXf &dZ) {
			Z = Z.array().tanh();
			dZ = Z.array() * (1 - Z.array()).array();
		}
		static float heaviside(float x) { return x > 0.0 ? 1.0 : x; }
		void relu(Eigen::MatrixXf &Z, Eigen::MatrixXf &dZ) {
			Z = Z.cwiseMax(0.0);
			dZ = Z.unaryExpr<float(*)(float)>(&heaviside);
		}
		
	public:
		virtual const float categorical_cross_entropy(const Eigen::MatrixXf &Y, float &accuracy) = 0;
		virtual const float mean_square_error(const Eigen::MatrixXf &Y, float &accuracy) = 0;
		virtual void forward(const Eigen::MatrixXf &X) = 0;
		virtual void update(const float &lr, const float &e) = 0;
		virtual void topology() = 0;
		virtual void reset() = 0;
		virtual void set_delta(const Eigen::MatrixXf &dL) = 0;
		
		virtual const Eigen::MatrixXf& getG() = 0;
		virtual const Eigen::MatrixXf& getW() = 0;
		virtual const Eigen::MatrixXf& getB() = 0;

		const Eigen::MatrixXf& getH() { return H; }
};

