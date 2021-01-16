#include "lstm.hpp"

template<const int Ns, const int Is, const int Os>
class Distributed_Dense : public Layer {
	protected:
		Eigen::MatrixXf gradB, gradW, oW, dH,\
		W = Eigen::MatrixXf::Constant(Is, Ns, 2/sqrtf(Is + Ns)), B = Eigen::MatrixXf::Ones(1, Ns);
        float vB = 0, vW = 0;

	public:
		Distributed_Dense(const std::string &&act = "sigmoid") { 
			activate = std::move(act); 
			H = Eigen::MatrixXf::Zero(Os, Ns);
		    dH = Eigen::MatrixXf::Zero(Os, Ns);	
			gradB = Eigen::MatrixXf::Zero(Os, Is);
		}
        void reset() { return; };
        void set_delta(const Eigen::MatrixXf &dL) { delta = dL.array() * dH.array(); }
        void topology() { std::cout << "W's shape: " << Is << " x " << Ns << " B's shape: " << Os << " x " << Ns << '\n';}
		void forward(const Eigen::MatrixXf &X) {                          
            if (X.cols() != W.rows()) { std::cout << "X's column size does not match W's row size\n"; }
            I = X;

            for (size_t w = 0; w < Os; w ++)
			{
				H.row(w) = I.row(w) * W + B;
				if (activate == "softmax") 
				{
					H.row(w) = (H.row(w).array() - H.row(w).maxCoeff()).array().exp();
					H.row(w) = H.row(w) / H.row(w).array().sum();
				}
			}
			
			if (activate == "relu") { 
				relu(H, dH);
			}else if (activate == "sigmoid") {
				sigmoid(H, dH);
			}else if (activate == "tanh") {
				ztanh(H, dH);
			}else { dH = H.array() * (1 - H.array()); }

        }
		void update(const float &lr, const float &e) {
			oW = W.transpose();
			I.transposeInPlace();
			for (size_t w = 0; w < Os; w ++)
			{
				gradW = (I.col(w) * delta.row(w));
				vW = 0.1 * vW + 0.9 * gradW.array().square().sum();
				vB = 0.1 * vW + 0.9 * delta.array().square().sum();
				W = W.array() - lr/sqrtf(vW + e) * gradW.array();
				B = B.array() - lr/sqrtf(vB + e) * delta.row(w).array();
				gradB.row(w) = delta.row(w) * oW;
			}
		}

		const Eigen::MatrixXf& getG() { return gradB; }
		const Eigen::MatrixXf& getW() { return W; }
		const Eigen::MatrixXf& getB() { return B; }

		const float categorical_cross_entropy(const Eigen::MatrixXf &Y, float &accuracy) {
            delta = H - Y;
            int acc = 0; float maxHc, maxYc;
            for (size_t j = 0; j < H.cols(); j ++)
            {
                maxHc = H.col(j).maxCoeff(); maxYc = Y.col(j).maxCoeff();
                for (size_t r = 0; r < H.col(j).size(); r ++)
                {
                    if (H.col(j)[r] == maxHc) { maxHc = r; break; }
                }
                for (size_t r = 0; r < Y.col(j).size(); r ++)
                {
					if (Y.col(j)[r] == maxYc) { maxYc = r; break; }
                }
				acc = maxHc == maxYc ? acc + 1 : acc;
            }
            acc = acc / Y.cols(); accuracy += acc;
            return (-Y.array() * H.array().log()).sum();
        }
 		const float mean_square_error(const Eigen::MatrixXf &Y, float &accuracy) {
              delta = (H - Y).array() * dH.array();
              accuracy = H.unaryExpr<float(*)(float)>(&roundf).isApprox(Y) ? accuracy + 1 : accuracy;
              return (0.5 * (Y - H).array().square()).sum();
        }
};	

template<const int Ns, const int Is, const int Os>
class Dense : public Layer {
	protected:
		Eigen::MatrixXf gradW, oW, dH, W = Eigen::MatrixXf::Constant(Is, Ns, 2/sqrtf(Is + Ns)), B = Eigen::MatrixXf::Ones(Os, Ns);
		float vB = 0, vW = 0;
	public:
		Dense(const std::string &&act = "sigmoid") { activate = std::move(act); }
		
		void reset() { return; };
		
		void set_delta(const Eigen::MatrixXf &dL) { delta = dL.array() * dH.array(); }

		void topology() { std::cout << "W's shape: " << Is << " x " << Ns << " B's shape: " << Os << " x " << Ns << '\n';}
		
		void forward(const Eigen::MatrixXf &X) {
			if (X.cols() != W.rows()) { std::cout << "X's column size does not match W's row size\n"; }
			I = X; H = X * W + B;
			if (activate == "relu") { Layer::relu(H, dH); } else if (activate == "hrelu") { Layer::hrelu(H, dH); }
			else if (activate == "tanh") { Layer::ztanh(H, dH); } else if (activate == "softmax") { Layer::softmax(H, dH); }
			else { Layer::sigmoid(H, dH); }
		}
		void update(const float &lr, const float &e) {
			oW = W;
			gradW = I.transpose() * delta;
			vW = 0.1 * vW + 0.9 * gradW.array().square().sum();
			vB = 0.1 * vB + 0.9 * delta.array().square().sum();
			W = W.array() - lr/sqrtf(vW + e) * gradW.array();
			B = B.array() - lr/sqrtf(vB + e) * delta.array();
			delta = delta * oW.transpose();
		}
		
		const Eigen::MatrixXf& getG() { return delta; }
		const Eigen::MatrixXf& getW() { return W; }
		const Eigen::MatrixXf& getB() { return B; }

		const float categorical_cross_entropy(const Eigen::MatrixXf &Y, float &accuracy) {
			delta = H - Y;
			int acc = 0; float maxHc, maxYc;
		    for (size_t j = 0; j < H.cols(); j ++)
			{
				maxHc = H.col(j).maxCoeff(); maxYc = Y.col(j).maxCoeff();
				for (size_t r = 0; r < H.col(j).size(); r ++)
				{
					if (H.col(j)[r] == maxHc) { maxHc = r; break; }
				}
				for (size_t r = 0; r < Y.col(j).size(); r ++)
				{
					if (Y.col(j)[r] == maxYc) { maxYc = r; break; }
				}
				acc = maxHc == maxYc ? acc + 1 : acc;
			}
			acc = acc / Y.cols(); accuracy += acc;
			return (-Y.array() * H.array().log()).sum();
		}
		const float mean_square_error(const Eigen::MatrixXf &Y, float &accuracy) {
			delta = (H - Y).array() * dH.array();
			accuracy = H.unaryExpr<float(*)(float)>(&roundf).isApprox(Y) ? accuracy + 1 : accuracy;
			return (0.5 * (Y - H).array().square()).sum();
		}
};
