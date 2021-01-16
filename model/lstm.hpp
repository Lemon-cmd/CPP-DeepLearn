#include "embedding.hpp" 

template <const int Ns, const int Is, const int Os>
class LSTM : public Layer {
	protected:
		Eigen::MatrixXf fo = Eigen::MatrixXf::Zero(Os, Ns), dfo = Eigen::MatrixXf::Zero(Os, Ns),
		io = Eigen::MatrixXf::Zero(Os, Ns), dio = Eigen::MatrixXf::Zero(Os, Ns),\ 
		oo = Eigen::MatrixXf::Zero(Os, Ns), doo = Eigen::MatrixXf::Zero(Os, Ns),\ 
		ca = Eigen::MatrixXf::Zero(Os, Ns), dca = Eigen::MatrixXf::Zero(Os, Ns),\ 
		cs = Eigen::MatrixXf::Zero(Os, Ns), dcs = Eigen::MatrixXf::Zero(Os, Ns),\
		gradFo, gradIo, gradCa, gradOo, oWf, oWi, oWo, oWca,\
		Wf = Eigen::MatrixXf::Constant(Is, Ns, 2/sqrtf(Ns + Is)), Wi = Eigen::MatrixXf::Constant(Is, Ns, 2/sqrtf(Ns + Is)),\
		Wo = Eigen::MatrixXf::Constant(Is, Ns, 2/sqrtf(Ns + Is)), Wca = Eigen::MatrixXf::Constant(Is, Ns, 2/sqrtf(Ns + Is)),\
		Uf = Eigen::MatrixXf::Constant(Ns, Ns, 2/sqrtf(Ns * 2)), Ui = Eigen::MatrixXf::Constant(Ns, Ns, 2/sqrtf(Ns * 2)),\
		Uo = Eigen::MatrixXf::Constant(Ns, Ns, 2/sqrtf(Ns * 2)), Uca = Eigen::MatrixXf::Constant(Ns, Ns, 2/sqrtf(Ns * 2)),\ 
		Bf = Eigen::MatrixXf::Ones(1, Ns), Bi = Eigen::MatrixXf::Ones(1, Ns),\ 
		Bo = Eigen::MatrixXf::Ones(1, Ns), Bca = Eigen::MatrixXf::Ones(1, Ns),\
		gradWf = Eigen::MatrixXf::Zero(Is, Ns), gradWi = Eigen::MatrixXf::Zero(Is, Ns),\
		gradWo = Eigen::MatrixXf::Zero(Is, Ns), gradWca = Eigen::MatrixXf::Zero(Is, Ns),\
		gradUf = Eigen::MatrixXf::Zero(Ns, Ns), gradUi = Eigen::MatrixXf::Zero(Ns, Ns),\
		gradUo = Eigen::MatrixXf::Zero(Ns, Ns), gradUca = Eigen::MatrixXf::Zero(Ns, Ns),\
		cp, hp, gradient; 
		
		float vWf = 0, vUf = 0, vBf = 0,\
			  vWi = 0, vUi = 0, vBi = 0,\
			  vWo = 0, vUo = 0, vBo = 0,\
			  vWca = 0, vUca = 0, vBca = 0;	
	public:
		LSTM() { 
			H = Eigen::MatrixXf::Zero(Os, Ns); 
			cp = Eigen::MatrixXf::Zero(Os, Ns);
			hp = Eigen::MatrixXf::Zero(Os, Ns);
			gradient = Eigen::MatrixXf::Zero(Os, Is);
		}

		const float categorical_cross_entropy(const Eigen::MatrixXf &Y, float &accuracy) {
			delta = -Y.array() / H.array(); if (H.row(0).maxCoeff() == Y.row(0).maxCoeff()) accuracy += 1;
			return (-Y.array() * H.array().log()).sum();
		}
		const float mean_square_error(const Eigen::MatrixXf &Y, float &accuracy) {
			delta = H - Y; if (H.isApprox(Y)) accuracy += 1;
			return (0.5 * (Y- H).array().square()).sum();
		}
		void forward(const Eigen::MatrixXf &X) {
			if (X.rows() != H.rows()) { std::cout << "Expected Sequence of length: " << Os << '\n'; return;}
			I = X;
			hp.row(0) = H.row(Os - 1);
			cp.row(0) = cp.row(Os - 1);

			for (size_t w = 0; w < Os; w ++)
			{
				fo.row(w) = (hp.row(w) * Uf + X.row(w) * Wf + Bf); 
				fo.row(w) = 1 / (1 + (-fo.row(w)).array().exp());
				dfo.row(w) = fo.row(w).array() * (1 - fo.row(w).array()).array();
					
				io.row(w) = (hp.row(w) * Ui + X.row(w) * Wi + Bi);
				io.row(w) = 1 / (1 + (-io.row(w)).array().exp());
				dio.row(w) = io.row(w).array() * (1 - io.row(w).array()).array();
				
				oo.row(w) = (hp.row(w) * Uo + X.row(w) * Wo + Bo); 
				oo.row(w) = 1 / (1 + (-oo.row(w)).array().exp());
				doo.row(w) = oo.row(w).array() * (1 - oo.row(w).array()).array();
				
				ca.row(w) = (hp.row(w) * Uca + X.row(w) * Wca + Bca);
				ca.row(w) = ca.row(w).array().tanh();
				dca.row(w) = ca.row(w).array() * (1 - ca.row(w).array()).array();
				
				cs.row(w) = fo.row(w).array() * cp.row(w).array() * io.row(w).array() * ca.row(w).array();
				cs.row(w) = cs.row(w).array().tanh();
				dcs.row(w) = cs.row(w).array() * (1 - cs.row(w).array()).array();
					
				H.row(w) = oo.row(w).array() * cs.row(w).array();	

				if (w < Os - 1) { cp.row(w + 1) = cs.row(w); hp.row(w + 1) = H.row(w); }
			}	
		}
		
		const Eigen::MatrixXf& getG() { return gradient; }
		const Eigen::MatrixXf& getW() { return Wf; }
		const Eigen::MatrixXf& getB() { return Bf; }

		void set_delta(const Eigen::MatrixXf& dL) { delta = dL; };
		
		void topology() { std::cout << "W's shape: " << Is << " x " << Ns << " B's shape: " << 1 << " x " << Ns << '\n';}
		
		void reset() { hp = Eigen::MatrixXf::Zero(Os, Ns); cp = Eigen::MatrixXf::Zero(Os, Ns); } 
		
		void update(const float &lr, const float &e) {
			/* Backpropagation through time */
			gradOo = delta.array() * cs.array() * doo.array();
			delta = delta.array() * oo.array() * dcs.array();
			gradIo = delta.array() * ca.array() * dio.array();
			gradFo = delta.array() * cp.array() * dfo.array();
			gradCa = delta.array() * io.array() * dca.array();
			
			I.transposeInPlace();
			Eigen::MatrixXf hp_t = hp.transpose();
			Eigen::MatrixXf Wo_t = Wo.transpose(), Wi_t = Wi.transpose(), Wf_t = Wf.transpose(), Wca_t = Wca.transpose();

			for (size_t w = 0; w < Os; w ++)
            {
				gradWo = gradWo + (I.col(w) * gradOo.row(w)).eval();	
				gradUo = gradUo + (hp_t.col(w) * gradOo.row(w)).eval();
				gradWi = gradWi + (I.col(w) * gradIo.row(w)).eval();
				gradUi = gradUi + (hp_t.col(w) * gradIo.row(w)).eval();
				gradWf = gradWf + (I.col(w) * gradFo.row(w)).eval();
				gradUf = gradUf + (hp_t.col(w) * gradFo.row(w)).eval();
				gradWca = gradWca + (I.col(w) * gradCa.row(w)).eval();
				gradUca = gradUca + (hp_t.col(w) * gradCa.row(w)).eval();
				gradient.row(w) = (gradOo.row(w) * Wo_t) + (gradIo.row(w) * Wi_t) + \
								  (gradFo.row(w) * Wf_t) + (gradCa.row(w) * Wca_t); 
			}
			
			gradWo = gradWo.array() / Os; 
			gradUo = gradUo.array() / Os; 
			gradOo = gradOo.colwise().sum().array() / Os;
			
			gradWca = gradWca.array() / Os; 
			gradUca = gradUca.array() / Os;
		   	gradCa = gradCa.colwise().sum().array() / Os;
			
			gradWi = gradWi.array() / Os; 
			gradUi = gradUi.array() / Os; 
			gradIo = gradIo.colwise().sum().array() / Os;
			
			gradWf = gradWf.array() / Os; 
			gradUf = gradUf.array() / Os;
		   	gradFo = gradOo.colwise().sum().array() / Os;

			vWf = 0.1 * vWf + 0.9 * gradWf.array().square().sum();
			vUf = 0.1 * vUf + 0.9 * gradUf.array().square().sum();
			vBf = 0.1 * vBf + 0.9 * gradFo.array().square().sum();
			
			vWi = 0.1 * vWi + 0.9 * gradWi.array().square().sum();
			vUi = 0.1 * vUi + 0.9 * gradUi.array().square().sum();
			vBi = 0.1 * vBi + 0.9 * gradIo.array().square().sum();

			vWo = 0.1 * vWo + 0.9 * gradWo.array().square().sum();
			vUo = 0.1 * vUo + 0.9 * gradUo.array().square().sum();
			vBo = 0.1 * vBo + 0.9 * gradOo.array().square().sum();

			vWca = 0.1 * vWca + 0.9 * gradWca.array().square().sum();
			vUca = 0.1 * vUca + 0.9 * gradUca.array().square().sum();
			vBca = 0.1 * vBca + 0.9 * gradCa.array().square().sum();

			Wo = Wo.array() - lr/sqrtf(vWo + e) * gradWo.array();
			Uo = Uo.array() - lr/sqrtf(vUo + e) * gradUo.array();
			Bo = Bo.array() - lr/sqrtf(vBo + e) * gradOo.array();
			
			Wf = Wf.array() - lr/sqrtf(vWf + e) * gradWf.array();
			Uf = Uf.array() - lr/sqrtf(vUf + e) * gradUf.array();
			Bf = Bf.array() - lr/sqrtf(vBf + e) * gradFo.array();
			
			Wi = Wi.array() - lr/sqrtf(vWi + e) * gradWi.array();
			Ui = Ui.array() - lr/sqrtf(vUi + e) * gradUi.array();
			Bi = Bi.array() - lr/sqrtf(vBi + e) * gradIo.array();

			Wca = Wca.array() - lr/sqrtf(vWca + e) * gradWca.array();
			Uca = Uca.array() - lr/sqrtf(vUca + e) * gradUca.array();
			Bca = Bca.array() - lr/sqrtf(vBca + e) * gradCa.array();	

			gradWf = gradWf.array() * 0; gradWi = gradWi.array() * 0; 
			gradWo = gradWo.array() * 0; gradWca = gradWca.array() * 0;
			gradUf = gradUf.array() * 0; gradUi = gradUi.array() * 0; 
			gradUo = gradUo.array() * 0; gradUca = gradUca.array() * 0;
		}
};
