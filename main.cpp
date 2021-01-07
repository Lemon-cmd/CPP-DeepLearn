#include "network.hpp"
#include "parser.hpp"
#include "parse_text.hpp" 

int main()
{
	//parse<75, 1> data ("./data/Tweets_txt.csv", "./data/Tweets_sentiment.csv");
	//std::cout << data.feature_size() << '\n';
	//parse_text<25, 20> data ("./data/sly_fox.txt");
	//std::cout << "Vocabulary size: " << data.vsize() << '\n';
	/*
	std::unique_ptr<network> net (new network("cross-entropy-error"));
	Layer* l0 = new Embedding<559, 1, 20>();
	Layer* l1 = new Embedding<100, 559, 20>();
	Layer* l2 = new LSTM<150, 100, 20>();
	Layer* l3 = new Dense<559, 150, 20>("softmax");
	*/

	parse_text<75, 20> data ("./data/news_0.txt"); std::cout << "Vocabulary size: " << data.vsize() << '\n';
	std::unique_ptr<network> net (new network());
	Layer* l0 = new Embedding<521, 1, 20>();
	Layer* l1 = new Embedding<100, 521, 20>();
	Layer* l2 = new LSTM<150, 100, 20>();
	Layer* l3 = new Dense<521, 150, 20>("softmax");

	net->add(l0); net->add(l1); net->add(l2); net->add(l3);

	net->show();
	net->train(data.get_Xtrain(), data.get_Ytrain(), 1000, 80);
	net->test(data.get_Xtest(), data.get_Ytest());
	net.reset();

}


