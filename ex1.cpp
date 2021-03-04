#include "./include/dpp/core.hpp"
#include "imgparser.hpp"

using namespace dpp;
#define L std::unique_ptr<Layer>
#define dim2d Eigen::DSizes<ptrdiff_t, 2>
#define dim3d Eigen::DSizes<ptrdiff_t, 3>

int main() {
	std::vector <Eigen::Tensor<float, 3>> X_train, Y_train, X_test, Y_test;

	load_mnist("./data/mnist/train-images.idx3-ubyte", 
			   "./data/mnist/train-labels.idx1-ubyte", 
			   X_train, Y_train);	

	load_mnist("./data/mnist/t10k-images.idx3-ubyte",
			   "./data/mnist/t10k-labels.idx1-ubyte",
			   X_test, Y_test);

	std::unique_ptr<Sequential> model {new Sequential()};
	model->add(L {new Conv2D (20, 2, 2, 2, 2)});
	model->add(L {new Conv2D (10, 2, 2, 1, 1)});
	model->add(L {new Conv2D (5, 2, 2, 1, 1)});
	model->add(L {new Pool2D (3, 3)});
	model->add(L {new Flat2D()});
	model->add(L {new Dense2D(10, 0.001, "softmax")});
	model->compile(dim3d {1, 28, 28}, 100, 10, "cross_entropy_error");
	model->summary();
	model->fit(X_train, Y_train);
	model->evaluate(X_test, Y_test);
	model->save_model("./models/mnist_model.txt");
	model.reset();
}
