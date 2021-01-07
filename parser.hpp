#include "modules.hpp"

template <const int train_split_percent, const int num_Ys>
class parse
{
	protected:
		int max_length = 0;
		std::vector<Eigen::MatrixXf> X_train, X_test, Y_train, Y_test;
		std::vector<Eigen::MatrixXf> Xdata, Ydata;

		const std::vector<float> tokenize(const std::string& line)
		{
			std::vector<float> tokens;
			float ascii = 0.0;
			int pos = 0;
			std::for_each(line.begin(), line.end(), [&tokens, &ascii, &pos](auto &ch) 
			{  
				if (ch != ' '){
					pos ++;
					ascii = ascii + (float)(unsigned int) ch * exp(-pos) + -pos * 3.14;
				}else if (ascii != 0.0 && ch == ' '){
					tokens.push_back(ascii);
					pos = 0; ascii = 0.0;
				}
			});
			
			tokens.push_back(ascii);
			max_length = tokens.size() > max_length ? tokens.size() : max_length;
			return tokens;
		}
		
		void split_train_test(const int &train_size)
		{
			int j; 
			X_train.resize(train_size); X_test.resize(Xdata.size() - train_size);
			Y_train.resize(train_size); Y_test.resize(Ydata.size() - train_size);
			for (j = 0; j < train_size; j ++) 
			{
				X_train[j] = Xdata[j]; Y_train[j] = Ydata[j];
			}
			for (j = 0; j < Xdata.size() - train_size; j ++)
			{
				X_test[j] = Xdata[j + train_size]; 
				Y_test[j] = Ydata[j + train_size]; 
			}
		}

	public:
		parse(const std::string &&Xpath, const std::string &&Ypath)
		{
			std::ifstream file(Xpath);
			std::string line;
			std::vector<std::vector<float>> X, Y;

			while (std::getline(file, line)) {
				if (!line.empty()) 
				{
					X.push_back(tokenize(line));
				}
			}
			file.close(); file = std::ifstream(Ypath);
			while (std::getline(file, line)) {
				if (!line.empty())
				{
					Y.push_back(tokenize(line));
				}
			}
			file.close();
			Xdata.resize(Y.size()); Ydata.resize(Y.size());
			std::cout << Xdata.size() << ' ' << Ydata.size() << '\n';
			for (int r = 0; r < Y.size(); r ++)
			{
				while (X[r].size() < max_length) { X[r].push_back(0.0); }
				Xdata[r] = Eigen::Map<Eigen::MatrixXf>(X[r].data(), max_length, 1);
				Ydata[r] = Eigen::Map<Eigen::MatrixXf>(Y[r].data(), num_Ys, 1);
			}
				
			int train_size = Xdata.size() * train_split_percent / 100;
			split_train_test(train_size);
		}
		
		const int feature_size()
		{
			return max_length;
		}	
		const std::vector<Eigen::MatrixXf>& get_Xtrain()
		{
			return X_train;
		}

		const std::vector<Eigen::MatrixXf>& get_Ytrain()
		{
			return Y_train;
		}

		const std::vector<Eigen::MatrixXf>& get_Xtest()
		{
			return X_test;
		}

		const std::vector<Eigen::MatrixXf>& get_Ytest()
		{
			return Y_test;
		}

};

