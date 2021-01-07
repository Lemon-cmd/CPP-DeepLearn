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
		virtual const float categorical_crossentropy(const Eigen::MatrixXf &Y, float &accuracy) = 0;
		virtual const float mean_square_error(const Eigen::MatrixXf &Y, float &accuracy) = 0;
		virtual void forward(const Eigen::MatrixXf &X) = 0;
		virtual void update(float &&lr = 0.001, float &&e=1e-8) = 0;
		virtual void topology() = 0;
		virtual void reset() = 0;
		virtual void set_delta() = 0;
		const Eigen::MatrixXf& getH() { return H; }
		Eigen::MatrixXf& getG() { return delta; }
};

template <const int Ns, const int Is, const int Fs>
class Embedding : public Layer {
	protected:
		Eigen::MatrixXf dH, oW, gradW, W = Eigen::MatrixXf::Constant(Ns, Is, expf(-(Ns + Is))), B = Eigen::MatrixXf::Ones(Ns, Fs);
		float vB, vW;
	public:
		Embedding() = default;
		void set_delta() { return; }
		void topology() { std::cout << "W's shape: " << Ns << " x " << Is << " B's shape: " << Ns << " x " << Fs << '\n';}
		void reset() { return; };
		void forward(const Eigen::MatrixXf &X) {
			I = X; oW = W;
			H = (W * X) + B;
			dH = I.transpose();
		}
		void update(float &&lr = 0.001, float &&e=1e-8) {
			 gradW = delta * dH; 
			 vW = 0.9 * gradW.array().square().sum();
			 vB = 0.9 * delta.array().square().sum();
			 W = W.array() - lr/sqrtf(vW + e) * gradW.array();
			 B = B.array() - lr/sqrtf(vB + e) * delta.array();
			 delta = oW.transpose() * delta;
		}
		const float categorical_crossentropy(const Eigen::MatrixXf &Y, float &accuracy) {
			delta = Y - H; if (H.row(0).maxCoeff() == Y.row(0).maxCoeff()) accuracy += 1;
			return (-Y.array() * H.array().log()).sum();
		}
		const float mean_square_error(const Eigen::MatrixXf &Y, float &accuracy) {
			delta = (H - Y).array() * dH.array(); if (H.unaryExpr<float(*)(float)>(&roundf).isApprox(Y)) accuracy += 1;
			return (0.5 * (Y - H).array().square()).sum();
		}
};

template <const int Ns, const int Is, const int Fs>
class LSTM : public Layer {
	protected:
		Eigen::MatrixXf fo, dfo, io, dio, oo, doo, ca, dca, gradFo, gradIo, gradCa, gradOo, cs, dcs, oWf, oUf, oWi, oUi, oWo, oUo, oWca, oUca,\
		Wf = Eigen::MatrixXf::Constant(Ns, Is, expf(-(Ns + Is))), Wi = Eigen::MatrixXf::Constant(Ns, Is, expf(-(Ns + Is))), Wo = Eigen::MatrixXf::Constant(Ns, Is, expf(-(Ns + Is))),\ 
		Wca = Eigen::MatrixXf::Constant(Ns, Is, expf(-(Ns + Is))), Uf = Eigen::MatrixXf::Constant(Ns, Ns, expf(-(Ns * 2))), Ui = Eigen::MatrixXf::Constant(Ns, Ns, expf(-(Ns * 2))),\ 
		Uo = Eigen::MatrixXf::Constant(Ns, Ns, expf(-(Ns * 2))), Uca = Eigen::MatrixXf::Constant(Ns, Ns, expf(-(Ns * 2))), Bf = Eigen::MatrixXf::Ones(Ns, Fs),\
		Bi = Eigen::MatrixXf::Ones(Ns, Fs), Bo = Eigen::MatrixXf::Ones(Ns, Fs), Bca = Eigen::MatrixXf::Ones(Ns, Fs),\ 
		hp = Eigen::MatrixXf::Zero(Ns, Fs), cp = Eigen::MatrixXf::Zero(Ns, Fs), ohp, ocp; 
	public:
		LSTM() { ohp = hp; ocp = cp; }
		void forward(const Eigen::MatrixXf &X) {
			I = X;
			fo = Uf * hp + Wf * I + Bf; sigmoid(fo, dfo);
			io = Ui * hp + Wi * I + Bi; sigmoid(io, dio);
			oo = Uo * hp + Wo * I + Bo; sigmoid(oo, doo);
			ca = Uca * hp + Wca * I + Bca; ztanh(ca, dca);
			cs = fo.array() * cp.array() + io.array() * ca.array(); ztanh(cs, dcs);
			H = oo.array() * cs.array();
			ohp = hp; ocp = cp;
			hp = H; cp = cs;
			oWf = Wf; oUf = Uf; oWi = Wi; oUi = Ui; oWo = Wo; oUo = Uo; oWca = Wca; oUca = Uca;  
		}
		const float categorical_crossentropy(const Eigen::MatrixXf &Y, float &accuracy) {
			delta = Y - H; if (H.row(0).maxCoeff() == Y.row(0).maxCoeff()) accuracy += 1;
			return (-Y.array() * H.array().log()).sum();
		}
		const float mean_square_error(const Eigen::MatrixXf &Y, float &accuracy) {
			delta = Y - H; if (H.isApprox(Y)) accuracy += 1;
			return (0.5 * (Y- H).array().square()).sum();
		}
		void set_delta() { return; };
		void topology() { std::cout << "W's shape: " << Ns << " x " << Is << " B's shape: " << Ns << " x " << Fs << '\n';}
		void reset() { hp = Eigen::MatrixXf::Zero(Ns, Fs); cp = Eigen::MatrixXf::Zero(Ns, Fs); } 
		void update(float &&lr=0.001, float &&e=1e-8) {
			gradOo = delta.array() * cs.array() * doo.array();
			Wo = Wo.array() - lr * (gradOo * I.transpose()).array();
			Uo = Uo.array() - lr * (gradOo * ohp.transpose()).array();
			Bo = Bo.array() - lr/sqrtf(0.9 * gradOo.array().square().sum() + e) * gradOo.array(); 	
	
			delta = delta.array() * oo.array() * dcs.array();
			gradIo = delta.array() * ca.array() * dio.array();
			gradFo = delta.array() * ocp.array() * dfo.array();
			gradCa = delta.array() * io.array() * dca.array();
			
			Wf = Wf.array() - lr * (gradFo * I.transpose()).array();
			Uf = Uf.array() - lr * (gradFo * ohp.transpose()).array();
			Bf = Bf.array() - lr/sqrtf(0.9 * gradFo.array().square().sum() + e) * gradFo.array(); 

			Wi = Wi.array() - lr * (gradIo * I.transpose()).array();
			Ui = Ui.array() - lr * (gradIo * ohp.transpose()).array();
			Bi = Bi.array() - lr/sqrtf(0.9 * gradIo.array().square().sum() + e) * gradIo.array(); 

			Wca = Wca.array() - lr * (gradCa * I.transpose()).array();
			Uca = Uca.array() - lr * (gradCa * ohp.transpose()).array();
			Bca = Bca.array() - lr/sqrtf(0.9 * gradCa.array().square().sum() + e) * gradCa.array(); 
			
			gradOo = oWo.transpose() * gradOo; gradFo = oWf.transpose() * gradFo;
			gradCa = oWca.transpose() * gradCa; gradIo = oWi.transpose() * gradIo;	
			delta = gradOo.array() * gradFo.array() * gradCa.array() * gradIo.array(); 
		}
};

template <const int Ns, const int Is, const int Fs>
class Dense : public Layer {
	protected:
		Eigen::MatrixXf dH, oW, gradW, W = Eigen::MatrixXf::Constant(Ns, Is, expf(-(Ns + Is))), B = Eigen::MatrixXf::Ones(Ns, Fs);
		float vB, vW;
	public:
		Dense(std::string &&act = "sigmoid") { activate = std::move(act); }
		void reset() { return; };
		void set_delta() { delta = delta.array() * dH.array(); }
		void topology() { std::cout << "W's shape: " << Ns << " x " << Is << " B's shape: " << Ns << " x " << Fs << '\n';}
		void forward(const Eigen::MatrixXf &X) {
			I = X; oW = W; H = W * X + B;
			if (activate == "relu") { Layer::relu(H, dH); } else if (activate == "hrelu") { Layer::hrelu(H, dH); } 
			else if (activate == "tanh") { Layer::ztanh(H, dH); } else if (activate == "softmax") { Layer::softmax(H, dH); } 
			else { Layer::sigmoid(H, dH); }
		}
		void update(float &&lr = 0.001, float &&e=1e-8) {
			gradW = delta * I.transpose();
			vW = 0.9 * gradW.array().square().sum();
			vB = 0.9 * delta.array().square().sum();
			W = W.array() - lr/sqrtf(vW + e) * gradW.array();
			B = B.array() - lr/sqrtf(vB + e) * delta.array();
			delta = oW.transpose() * delta;
		}
		const float categorical_crossentropy(const Eigen::MatrixXf &Y, float &accuracy) {
			delta = Y - H;
			int acc = 0;
		    	for (size_t j = 0; j < H.cols(); j ++)
			{
				acc = H.col(j).maxCoeff() == Y.col(j).maxCoeff() ? acc + 1 : acc;
			}      
			acc = acc / Y.cols(); accuracy += acc;	
			return (-Y.array() * H.array().log()).sum();
		}
		const float mean_square_error(const Eigen::MatrixXf &Y, float &accuracy) {
			delta = (H - Y).array() * dH.array(); if (H.unaryExpr<float(*)(float)>(&roundf).isApprox(Y)) accuracy += 1;
			return (0.5 * (Y - H).array().square()).sum();
		}
};
