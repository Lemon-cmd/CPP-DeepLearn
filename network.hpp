#include "layer.hpp" 

class network {
	protected:
		std::vector<Layer*> net; 
		float loss = 0.0; float accuracy = 0.0; int _size = 0;
		int j, i; std::string cost_fun;
		void forward(const Eigen::MatrixXf &X, const Eigen::MatrixXf &Y) {
			for (j = 0; j < _size; j ++) {
				if (j == 0) { net[j]->forward(X); }
				else { net[j]->forward(net[j - 1]->getH()); }
			}
			loss = cost_fun == "cross_entropy_error" ? net.back()->categorical_crossentropy(Y, accuracy) : net.back()->mean_square_error(Y, accuracy);	
		}
		void backward() {
			net.back()->update();
			for (j = _size - 2; j > -1; j--) {
				net[j]->getG() = net[j + 1]->getG();
				net[j]->set_delta();
				net[j]->update();
			}
		}
		void reset_network() {
			std::for_each(net.begin(), net.end(), [](auto &L){L->reset();});
		}
		const int rand_num(const int &start, const int &end) {
			static std::random_device rd; static std::mt19937_64 gen(rd());
			std::uniform_int_distribution<int> dist(start, end);
			return dist(gen);
		}
		
	public:
		network(std::string &&cf = "mean_square_error") { cost_fun = std::move(cf); }
		~network() { 
			if (_size > 0) std::for_each(net.begin(), net.end(), [](auto &L){ delete L; });
			net.clear();
		}
		void add(Layer* &layer) {
			net.push_back(std::move(layer)); _size = _size + 1;
		}
		void show() { if (_size == 0) { return; } else { std::for_each(net.begin(), net.end(), [](auto &L){ L->topology();} ); }}
		
		void train(const std::vector<Eigen::MatrixXf> &X, const std::vector<Eigen::MatrixXf> &Y, const int &&epochs=1000, const int &&batch_size=100) {
			assert(X.size() == Y.size());
			const int data_size = X.size();
			int e, start = 0; int end = start + batch_size; int size = end - start; int half = start + (size / 2), su, se;
			su = rand_num(start, half); se = rand_num(half, end);
			for (e = 0; e <= epochs; e++) {
				for (i = start; i < end; i++) {
					forward(X[i], Y[i]);
					if (i >= su && i <= se) backward();
				}
				if (e % 100 == 0) { std::cout << "Epoch: " << e << " Accuracy: " << accuracy/size * 100 << "% Loss: " << loss << '\n'; }
				start += batch_size; end += batch_size;
				start = end >= data_size ? 0 : start;
				end = end >= data_size ? start + batch_size : end;
				size = end - start; half = start + size / 2;
				su = rand_num(start, half); se = rand_num(half, end);
				accuracy = 0.0;
			}
			accuracy = 0.0;
			reset_network();
		}
		
		void test(const std::vector<Eigen::MatrixXf> &X, const std::vector<Eigen::MatrixXf> &Y) {
			if (X.size() != Y.size()) return;
			std::cout << "Testing model...\n";
			for (i = 0; i < X.size(); i++) {
				forward(X[i], Y[i]);
			}
			std::cout << "Test accuracy: " << accuracy/X.size() * 100 << " Loss: " << loss << '\n';
			accuracy = 0.0;
			reset_network();
		}	
};
