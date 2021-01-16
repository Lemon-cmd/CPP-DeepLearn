#include "./model/network.hpp"


#define L std::unique_ptr<Layer>

int main()
{
	parse_text<75, 10> data ("./data/sly_fox.txt");
	network net("categorical_cross_entropy");
	net.add(L {new embedding<560, 50, 10>()});
	net.add(L {new LSTM<100, 50, 10>()});
	net.add(L {new Distributed_Dense<560, 100, 10>("softmax")});

	net.compile(50000, 0); //epoch, batch, learning rate, optimization rate
	net.fit(data.get_X(), data.get_Y());
}
