#include "modules.hpp"

template <const int train_percent, const int sequence_length>
class parse_text
{
    protected:
		std::vector<Eigen::MatrixXf> X_train, X_test, Y_train, Y_test, X, Y;
		std::map<std::string, int> key_id;
		std::map<int, std::string> id_key;
		int id = 3;
	
		void tokenize(const std::string& line, std::vector<float> &tokens)
        {   
			std::string word = "";
			std::for_each(line.begin(), line.end(), [this, &tokens, &word](const auto &ch)
			{  
				const float ascii = (unsigned int) ch;
				if ( (ascii >= 48 && ascii <= 57) || (ascii >= 65 && ascii <= 90) || (ascii >= 97 && ascii <= 122) ){
					word = word + ch;
				}else if (word != "") {
					if (key_id.find(word) == key_id.end())
					{   
						key_id[word] = id; id_key[id] = word; id ++;
					}
					tokens.push_back(key_id[word]); word = "";
					if (ch != ' ')
					{   
						word += ch;
						if (key_id.find(word) == key_id.end())
						{   
							key_id[word] = id; id_key[id] = word; id ++;
						}
						tokens.push_back(key_id[word]); word = "";
					}
				}
			});
		}
		
		void process_X(const int &size, const std::vector<float> &corpus, std::vector<Eigen::MatrixXf> &X)
		{
			int j, c = 0;
			for (j = 0; j < size; j += sequence_length) 
			{
				std::vector<float> Xrow (corpus.begin() + j, corpus.begin() + j + sequence_length);
				X[c] = Eigen::Map<Eigen::MatrixXf>(Xrow.data(), 1, sequence_length);
				c ++;
			}
			if (j < corpus.size())
			{
				std::vector<float> Xrow(corpus.begin() + j, corpus.end());
			    int L = Xrow.size();
				while (L < sequence_length) { Xrow.push_back(0.0); L ++; };
				X[c] = Eigen::Map<Eigen::MatrixXf>(Xrow.data(), 1, sequence_length);
			}
		}

		void process_Y(const int &size, const std::vector<float> &corpus, std::vector<Eigen::MatrixXf> &Y)
		{
			int j, i, L, c = 0;
			for (j = 0; j < size; j += sequence_length)
			{
				i = j; L = 0;	
				Eigen::MatrixXf Yrow(id, sequence_length);
				while (L < sequence_length)
				{
					std::vector<float> v(id, 0); v[corpus[i]] = 1;
					Yrow.col(L) = Eigen::Map<Eigen::MatrixXf>(v.data(), id, 1);
					i ++;
					L ++;
				}
				Y[c] = Yrow; c ++;
			}
			if (j < corpus.size())
			{
				Eigen::MatrixXf Yrow(id, sequence_length);
				i = j; L = 0;
				while (i < corpus.size())
				{
					std::vector<float> v(id, 0); v[corpus[i]] = 1;
					Yrow.col(L) = Eigen::Map<Eigen::MatrixXf>(v.data(), id, 1);
					i ++; L ++;
				}
				while (L < sequence_length) 
				{
					std::vector<float> v(id, 0); v[0] = 1;
				   	Yrow.col(L) = Eigen::Map<Eigen::MatrixXf>(v.data(), id, 1);
					L ++;
				}	
				Y[c] = Yrow;	
			}
		}
		
		void split_train_test(const int &train_size)
		{
			int j;
			X_train.resize(train_size); X_test.resize(X.size() - train_size);
			Y_train.resize(train_size); Y_test.resize(Y.size() - train_size);
			for (j = 0; j < train_size; j ++)
			{
				X_train[j] = X[j]; Y_train[j] = Y[j];
			}
			for (j = 0; j < X_test.size(); j ++)
			{
				X_test[j] = X[j + train_size];
				Y_test[j] = Y[j + train_size];
			}
		}

	public:
		parse_text(const std::string &&path, const bool &&split=false)
        {
			key_id["-OOV-"] = 1; key_id["-PAD-"] = 0; key_id["\n"] = 2; id_key[0] = "-PAD-"; id_key[1] = "-OOV-"; id_key[2] = "\n";
			std::ifstream file(std::move(path));
            std::string line;
            std::vector<float> corpus;
			
			while (std::getline(file, line)) {
				if (!line.empty())
				{   
					tokenize(line, corpus);
					corpus.push_back(key_id["\n"]);
				}
			}
			file.close();
	
			const int data_size = corpus.size() - sequence_length;
			X.resize(data_size / sequence_length + 2); Y.resize(data_size / sequence_length + 2);
			std::cout << "Processing X...\n";
			process_X(data_size, corpus, X);
			std::cout << "Processing Y...\n";
			process_Y(data_size, corpus, Y);
			std::cout << "Size: " << id << '\n';
			if (split) { const int train_size = X.size() * train_percent / 100; split_train_test(train_size); }
			else { X_train = X; Y_train = Y; X_test = X; Y_test = Y;}
		}

		const auto& getX() { return X; }
		const auto& getY() { return Y; }
		const auto& vsize() { return id; }
		const auto& get_Xtrain() { return X_train; }
		const auto& get_Ytrain() { return Y_train; }
		const auto& get_Xtest() { return X_test; }
		const auto& get_Ytest() { return Y_test; }
};
