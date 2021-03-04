#include "./include/dpp/core.hpp"

using namespace dpp;
#define L std::unique_ptr<Layer>
#define dim2d Eigen::DSizes<ptrdiff_t, 2>
#define dim3d Eigen::DSizes<ptrdiff_t, 3>

int main() {

	//A class for parsing a corpus  	
	//ParseText<50, 50> corpus ("./data/sly_fox.txt");
	
	// you can create a model and manually add new layers
	
	/*
	std::unique_ptr<Sequential> model {new Sequential()};

	model->add(L {new Embedding (corpus.GetVocabSize(), 200}));
	model->add(L {new Embedding (corpus.GetVocabSize(), 200}));
	model->add(L {new LSTM(100)});
	model->add(L {new DistributedDense(corpus.GetVocabSize(), 0.001, "softmax")});

	model->compile(dim2d {50, 1}, 100, 100, "cross_entropy_error");		// compile the model: input_shape, epochs, batch_size, cost function
	model->fit(corpus.GetSetX(), corpus.GetSetY());						// train
	model->evaluate(corpus.GetSetX(), corpus.GetSetY());				// test

	*/
	
	/*
	std::unique_ptr<Sequential> model {new Sequential()};				// or you can load a pretrained model
	model->load_model("./models/sly_model.txt");		
	*/
	
	// Additionally, you can manually create layers outside of sequential
	// L l0 { new Dense(100) };
	// but you are required to initialize its shape
	// l0->init(dim2d { 1, 10} ); 
	// you can also load pretrained weights
	// l0->load(dim2d {1, 10}, dim2d {1, 100}, std::vector <float> (100, 0.0), std::vector <float> (100 * 10, 0.0)
	// input shape, output shape, bias, weight
}
