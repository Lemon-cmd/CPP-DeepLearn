#include "layer.hpp" 

template <const int Ns, const int Is, const int Os>
class embedding : public Layer {
	protected:
		Eigen::MatrixXf gradient, gradW, oW, W = Eigen::MatrixXf::Constant(Ns, Is, 2/sqrtf(Ns + Is));	
		float vW = 0;
	public:
		embedding() { 
			H = Eigen::MatrixXf::Zero(Os, Is); 
			gradient = Eigen::MatrixXf::Zero(Os, 1);
		}
		
		void set_delta(const Eigen::MatrixXf &dL) { delta = dL; }
		void topology() { std::cout << "Embedding Layer \tWeight shape: " << Ns << " x " << Is << "\tOutput shape: " << Os << " x " << Is << '\n';}
		void reset() { return; };
		void forward(const Eigen::MatrixXf &X) {
			if (X.rows() != Os) { std::cout << "Input length does not match the designated sequence length"; }
			I = X; int ind;
			for (size_t w = 0; w < Os; w ++)
			{
				ind = I.coeff(w, 0) >= 0 ? I.coeff(w, 0) : 0;
				H.row(w) = I.row(w) * W.row(ind);	
			}
		}
		void update(const float &lr, const float &e) {
			oW = W;
			int ind;	
			for (size_t w = 0; w < Os; w ++)
			{
				ind = I.coeff(w, 0) >= 0 ? I.coeff(w, 0) : 0;
				gradW = (I.row(w) * delta.row(w)).array();
				vW = 0.1 * vW + 0.9 * gradW.array().square().sum();
				
				W.row(ind) = W.row(ind).array() - lr/sqrtf(vW + e) * gradW.array(); 
				gradient.row(w) = oW.row(ind) * delta.row(w).transpose();
			}
		}

		const Eigen::MatrixXf& getG() { return gradient; }
		const Eigen::MatrixXf& getW() { return W; }
		const Eigen::MatrixXf& getB() { return W; }

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
			delta = H - Y;
			accuracy = H.unaryExpr<float(*)(float)>(&roundf).isApprox(Y) ? accuracy + 1 : accuracy;
			return (0.5 * (Y - H).array().square()).sum();
		}

};
