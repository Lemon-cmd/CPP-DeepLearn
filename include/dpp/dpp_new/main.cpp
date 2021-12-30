#include "core.hpp"

using namespace dpp;

#define L std::unique_ptr<Layer>
#define Tensor3d Eigen::Tensor<float, 3>
#define Tensor4d Eigen::Tensor<float, 4>
#define Dim1d Eigen::DSizes<ptrdiff_t, 1>
#define Dim2d Eigen::DSizes<ptrdiff_t, 2>
#define Dim3d Eigen::DSizes<ptrdiff_t, 3>
#define Dim4d Eigen::DSizes<ptrdiff_t, 4>


int main() {
	/*
	Tensor3d X {3, 150, 150}; X.setRandom();

	L l0 (new Conv2D {5, 3, 3, 2, 2, "relu"});
	L l1 (new Conv2D {5, 3, 3});
	L l2 (new Conv2D {3, 3, 3});
	//L l3 (new Flatten {});
	
	l0->init(X.dimensions());
	l1->init(l0->Get3dOutputDim());
	l2->init(l1->Get3dOutputDim());

	float accuracy = 0.0;
	Tensor3d Y {l2->Get3dOutputDim()}; Y.setRandom();
	
	for (int e = 0; e < 50; e ++) {
		l0->Forward(X);
		l1->Forward(l0->Get3dOutput());
		l2->Forward(l1->Get3dOutput());
		std::cout << "J: " << l2->MeanSquaredError(Y, accuracy) << '\n';;
		l2->Update();
		l1->SetDelta(l2->Get3dDelta());
		l1->Update();
		l0->SetDelta(l1->Get3dDelta());
		l0->Update();
	}
	*/
	
	Eigen::MatrixXf X = Eigen::MatrixXf::Random(50, 1), Y = Eigen::MatrixXf::Random(50, 10);
		
	L l0 (new Embedding {2, 10});
	L l1 (new LSTM {5, "sigmoid"});
	L l2 (new LSTM {10, "sigmoid"});
	
	l0->init(Dim2d{50, 1});
	l1->init(l0->Get2dOutputDim());
	l2->init(l1->Get2dOutputDim());
	
	float accuracy = 0.0;
	for (int e = 0; e < 500; e ++) {
		l0->Forward(X);
		cout << l0->Get2dOutput().rows() << ' ' << l0->Get2dOutput().cols() << '\n';
		l1->Forward(l0->Get2dOutput());
		l2->Forward(l1->Get2dOutput());
		cout << "J: " << l2->MeanSquaredError(Y, accuracy) << '\n';;
		l2->Update();
		l1->SetDelta(l2->Get2dDelta());
		l1->Update();
		l0->SetDelta(l1->Get2dDelta());
		l0->Update();
	}
	
}
