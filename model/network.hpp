#include "dense.hpp" 
#include "parse_text.hpp" 
#include <assert.h>
#include <iomanip>

class network
{
	protected:
		float loss = 0.0, accuracy = 0.0, lrate, erate;
		int size = 0, batch, epoch, i;
		bool compile_flag = false, batch_flag = false;
		std::vector<std::unique_ptr<Layer>> net;  
		std::string cost_fun;
		
		const int randint (const int min, const int max)
		{
			std::random_device rd;
			std::mt19937_64 gen(rd());
			std::uniform_int_distribution<int> dist(min, max - 1);
			return dist(gen); 
		}
		
		void forward(const Eigen::MatrixXf &X, const Eigen::MatrixXf &Y)
		{
			net[0]->forward(X);
			for (i = 1; i < size; i ++)
			{
				net[i]->forward(net[i - 1]->getH());
			}
			loss = cost_fun == "mean_square_error" ? net[size - 1]->mean_square_error(Y, accuracy) : \
				   net[size - 1]->categorical_cross_entropy(Y, accuracy);
		}

		void backward()
		{
			net.back()->update(lrate, erate);
			for (i = size - 2; i >= 0; i --)
			{
				net[i]->set_delta(net[i + 1]->getG());
				net[i]->update(lrate, erate);
			}
		}

		void test(const std::vector<Eigen::MatrixXf> &X, const std::vector<Eigen::MatrixXf> &Y)
		{
			assert(X.size() == Y.size());
			for (int j = 0; j < X.size(); j ++)
			{
				forward(X[j], Y[j]);
			}
			std::cout << "Loss: " << loss << " Accuracy: " << accuracy * 100 / X.size() << '\n';
			accuracy = 0.0;
		}

	public:
		network(std::string &&cf = "mean_square_error") 
		{
			assert(cf == "mean_square_error" || cf == "categorical_cross_entropy");	
			cost_fun = std::move(cf);
		}
		~network()
		{
			if (size > 0)
			{
				std::for_each(net.begin(), net.end(), [](auto &L) { L.reset(); });
			}
		}
			
		void add(std::unique_ptr<Layer> &&L)
		{
			net.emplace_back(std::move(L));
			size += 1;
		}
		
		void compile(const int &&epoch_size = 100, const int &&batch_size = 0, const float &&lr = 0.001, const float &&er = 1e-8)
		{
			assert(epoch_size > 0);
			batch_flag = batch_size != 0 ? true : batch_flag;
			compile_flag = true;
			erate = std::move(er);
			lrate = std::move(lr);
			epoch = std::move(epoch_size); 
			batch = std::move(batch_size);
		}

		void fit(const std::vector<Eigen::MatrixXf> &X, const std::vector<Eigen::MatrixXf> &Y)
		{
			assert(X.size() == Y.size() && compile_flag == true);
			batch = batch_flag ? batch : X.size() * 0.10;
			int start = 0, end = batch, limitS = X.size() - batch, limitE = X.size();
			int half = batch/2;
			int su = randint(start, half), se = randint(half, end);	

			for (int e = 0; e < epoch; e ++)
			{
				for (int j = start; j < end; j ++)
				{
					forward(X[j], Y[j]);
					//if (j >= su && j <= se)
					//{
						backward();	
					//}	
				}
				
				if (e % 100 == 0) { std::cout << "Epoch: " << e << " Loss: " << loss << " Accuracy: " << accuracy * 100 / (end - start) << '\n';}
				
				start = start + batch;
				end = end + batch;	
				start = start <= limitS ? start : 0;
				end = end <= limitE ? end : batch;
				half = (start + end)/2;
				su = randint(start, half);
				se = randint(half, end);	
				accuracy = 0.0;
			}	
			accuracy = 0.0;
			test(X, Y);
		}
		

};
