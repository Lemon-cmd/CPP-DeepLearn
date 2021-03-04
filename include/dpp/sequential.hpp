namespace dpp {

class Sequential {
	public:
		Sequential() { std::cout << std::fixed << std::setprecision(8); }
		
		/* 
		 * Destructor 
		 * If there are layers, reset their pointer and clear the vector.
		*/

		~Sequential() { 
			if (size_ > 0) { 
				std::for_each(network_.begin(), network_.end(), [](auto &layer) { layer.reset(); });
				network_.clear();
			} 
		}

		/* 
		 * List the information of Model 
		*/

		void summary() {
			assert(size_ > 0);
			if (!compile_flag_ && !reload_flag_) { std::cout << "The Model is not compiled.\n\n"; }
			std::for_each(network_.begin(), network_.end(), [](auto &layer) { layer->topology(); });
			std::cout << '\n';
		}
		
		/* 
		 * Add Layer Method (Take Reference or L-Value)
		 * Ensure that compilation of the model had not happened. 
		*/

		void add(std::unique_ptr<Layer> &&L) {
			assert(!compile_flag_ && !reload_flag_);

			// set the appropriate flag type
			matrix_flag_ = L->type() == "2D" ? true : matrix_flag_;
			tensor_3d_flag_ = L->type() == "3D" ? true : tensor_3d_flag_;
			tensor_4d_flag_ = L->type() == "4D" ? true : tensor_4d_flag_;
			
			// ensure that all layers are of the same type
			assert(!((matrix_flag_ && tensor_3d_flag_ && tensor_4d_flag_) || (matrix_flag_ && tensor_3d_flag_ && !tensor_4d_flag_) || 
					(matrix_flag_ && !tensor_3d_flag_ && tensor_4d_flag_) || (!matrix_flag_ && tensor_3d_flag_ && tensor_4d_flag_)));

			// add to network (move unique pointer)
			network_.push_back(std::move(L));
			size_ ++; // increment size of network
		}

		void add(std::unique_ptr<Layer> &L) {
			assert(!compile_flag_ && !reload_flag_);

			// set the appropriate flag type
			matrix_flag_ = L->type() == "2D" ? true : matrix_flag_;
			tensor_3d_flag_ = L->type() == "3D" ? true : tensor_3d_flag_;
			tensor_4d_flag_ = L->type() == "4D" ? true : tensor_4d_flag_;
			
			// ensure that all layers are of the same type
			assert(!((matrix_flag_ && tensor_3d_flag_ && tensor_4d_flag_) || (matrix_flag_ && tensor_3d_flag_ && !tensor_4d_flag_) || 
					(matrix_flag_ && !tensor_3d_flag_ && tensor_4d_flag_) || (!matrix_flag_ && tensor_3d_flag_ && tensor_4d_flag_)));

			// add to network (move unique pointer)
			network_.push_back(std::move(L));
			size_ ++; // increment size of network
		}

		/* 
		 * Compilation Method (Take 2D, 3D, and 4D input shape)
		 * Ensure that all of the required values are above 0.
		 * Set the metadata
		 * Set the input of layers in the network
		*/

		void compile(const Eigen::DSizes<ptrdiff_t, 2> input_shape, const int epochs = 1, const int batch_size = 10, const std::string cost_function = "mean_squared_error") {
			assert(!reload_flag_ && epochs > 0 && batch_size > 0 && size_ > 0 && input_shape[0] > 0 && input_shape[1] > 0);
			compile_flag_ = true;
			epochs_ = epochs;
			batch_size_ = batch_size;
			SetResetVector();
			SetLayersInput(input_shape);
			cost_function_ = cost_function == "mean_squared_error" ? cost_function : "cross_entropy_error";
		}
	
		void compile(const Eigen::DSizes<ptrdiff_t, 3> input_shape, const int epochs = 1, const int batch_size = 10, const std::string cost_function = "mean_squared_error") {
			assert(!reload_flag_ && epochs > 0 && batch_size > 0 && size_ > 0 && input_shape[0] > 0 && input_shape[1] > 0 && input_shape[2] > 0);
			compile_flag_ = true;
			epochs_ = epochs;
			batch_size_ = batch_size;
			SetResetVector();
			SetLayersInput(input_shape);
			cost_function_ = cost_function == "mean_squared_error" ? cost_function : "cross_entropy_error";
		}
		
		void compile(const Eigen::DSizes<ptrdiff_t, 4> input_shape, const int epochs = 1, const int batch_size = 10, const std::string cost_function = "mean_squared_error") {
			assert(!reload_flag_ && epochs > 0 && batch_size > 0 && size_ > 0 && input_shape[0] > 0 && input_shape[1] > 0 && input_shape[2] > 0);
			compile_flag_ = true;
			epochs_ = epochs;
			batch_size_ = batch_size;
			SetResetVector();
			SetLayersInput(input_shape);
			cost_function_ = cost_function == "mean_squared_error" ? cost_function : "cross_entropy_error";
		}

		/* Train Method (Take a vector of 2D, 3D, and 4D data)
		 * Ensure that compilation had been done, size of training data (X & Y) are equal, and the appropriate data type flag.
		 * Perform training over the set of epochs and batches of data.
		 * Print the loss and accuracy per epoch.
		*/

		void fit(const std::vector<Eigen::MatrixXf> &X_train, const std::vector<Eigen::MatrixXf> &Y_train) {
			assert((reload_flag_ || compile_flag_) && X_train.size() == Y_train.size() && matrix_flag_);
			const int train_size = X_train.size();

			for (int e = 0; e < epochs_; e ++) {
				for (int i = 0; i < train_size; i ++) {
					Forward(X_train[i], Y_train[i]);
					
					if (rand() > 0.5) {
						Backward2D();
					}
				}
				
				std::cout << "Epoch: " << e + 1 << "\tLoss: " << loss_ / train_size << "\tAccuracy: " << accuracy_ / train_size<< '\n';
				accuracy_ = 0.0; loss_ = 0.0;
			}	
		}

		void fit(const std::vector<Eigen::Tensor<float, 3>> &X_train, const std::vector<Eigen::Tensor<float, 3>> &Y_train) {
            assert((reload_flag_ || compile_flag_) && X_train.size() == Y_train.size() && tensor_3d_flag_);
			const int train_size = X_train.size();

			for (int e = 0; e < epochs_; e ++) {
				for (int i = 0; i < train_size; i ++) {
					Forward(X_train[i], Y_train[i]);
					
					if (rand() > 0.5) {
						Backward3D();
					}
				}	
				
				std::cout << "Epoch: " << e + 1 << "\tLoss: " << loss_ / train_size << "\tAccuracy: " << accuracy_ / train_size << '\n';
				accuracy_ = 0.0; loss_ = 0.0;
			}	
        }

		void fit(const std::vector<Eigen::Tensor<float, 4>> &X_train, const std::vector<Eigen::Tensor<float, 4>> &Y_train) {
			assert((reload_flag_ || compile_flag_) && X_train.size() == Y_train.size() && tensor_4d_flag_);
			const int train_size = X_train.size();

			for (int e = 0; e < epochs_; e ++) {
				for (int i = 0; i < train_size; i ++) {
					Forward(X_train[i], Y_train[i]);

					if (rand() > 0.5) {
						Backward4D();
					}
				}	

				std::cout << "Epoch: " << e + 1 << "\tLoss: " << loss_ / train_size << "\tAccuracy: " << accuracy_ / train_size << '\n';
				accuracy_ = 0.0; loss_ = 0.0;	
			}	
        } 

		/*
		 *
		 * Fit Sequence Method
		 * Only use this when the network contains sequence layer(s) (e.g., LSTM or RNN) 
		 *
		 */

		void fit_sequence(const std::vector<Eigen::MatrixXf> &X_train, const std::vector<Eigen::MatrixXf> &Y_train) {
			assert((reload_flag_ || compile_flag_) && X_train.size() == Y_train.size() && matrix_flag_);
			const int train_size = X_train.size();

			for (int e = 0; e < epochs_; e ++) {
				for (int i = 0; i < train_size; i ++) {
                    Forward(X_train[i], Y_train[i]);

					if (rand() > 0.3) {
						Backward2D();
					}
                }

				// reset recurrent parameters
				std::for_each(reset_indexes_.begin(), reset_indexes_.end(), [this](auto &index) { network_[index]->reset(); });

                std::cout << "Epoch: " << e + 1 << "\tLoss: " << loss_ / train_size << "\tAccuracy: " << accuracy_ / train_size<< '\n';
                accuracy_ = 0.0; loss_ = 0.0;
            }
		}

		/* Evaluation Method for determining the performance of the model
		 * Similar to the fit method, parameters come in three forms.
		 * Ensure that the data vectors are of the same size and the data type flag is checked.
		 * Perform only forward propagation.
		*/

		void evaluate(const std::vector<Eigen::MatrixXf> &X_test, const std::vector<Eigen::MatrixXf> &Y_test) {
			assert((reload_flag_ || compile_flag_) && X_test.size() == Y_test.size() && matrix_flag_);
			
			for (int i = 0; i < X_test.size(); i ++) {
				Forward(X_test[i], Y_test[i]);
			}	

			std::cout << "Loss: " << loss_ << "\tAccuracy: " << accuracy_ / X_test.size() * 100 << '\n'; 
			accuracy_ = 0.0;
		}

		void evaluate(const std::vector<Eigen::Tensor<float, 3>> &X_test, const std::vector<Eigen::Tensor<float, 3>> &Y_test) {
			assert((reload_flag_ || compile_flag_) && X_test.size() == Y_test.size() && tensor_3d_flag_);
			
			for (int i = 0; i < X_test.size(); i ++) {
				Forward(X_test[i], Y_test[i]);
			}	

			std::cout << "Loss: " << loss_ << "\tAccuracy: " << accuracy_ / X_test.size() * 100 << '\n'; 
			accuracy_ = 0.0;
		}
			
		void evaluate(const std::vector<Eigen::Tensor<float, 4>> &X_test, const std::vector<Eigen::Tensor<float, 4>> &Y_test) {
			assert((reload_flag_ || compile_flag_) && X_test.size() == Y_test.size() && tensor_4d_flag_);
			
			for (int i = 0; i < X_test.size(); i ++) {
				Forward(X_test[i], Y_test[i]);
			}	

			std::cout << "Loss: " << loss_ << "\tAccuracy: " << accuracy_ / X_test.size() * 100 << '\n'; 
			accuracy_ = 0.0;
		}
		
		/* Predict Method 
		 * Perform Forward Propagation on a single entry
		 * Output the prediction
		*/

		void predict(const Eigen::MatrixXf &X, const Eigen::MatrixXf &Y) {
			assert(size_ > 0);
			Forward(X, Y);
			std::cout << "Loss: " << loss_ << "\tAccuracy: " << accuracy_ * 100 << '\n';
			accuracy_ = 0.0;
		}

		void predict(const Eigen::Tensor<float, 3> &X, const Eigen::Tensor<float, 3> &Y) {
			assert(size_ > 0);
			Forward(X, Y);
			std::cout << "Loss: " << loss_ << "\tAccuracy: " << accuracy_ * 100 << '\n';
			accuracy_ = 0.0;
		}
		
		void predict(const Eigen::Tensor<float, 4> &X, const Eigen::Tensor<float, 4> &Y) {
			assert(size_ > 0);
			Forward(X, Y);
			std::cout << "Loss: " << loss_ << "\tAccuracy: " << accuracy_ * 100 << '\n';
			accuracy_ = 0.0;
		}

		/* 
		 * Save Model Method
		*/
		
		void save_model(const std::string file_path) {
			assert(compile_flag_ || reload_flag_);
			std::ofstream file;

			// open file
			file.open(file_path);

			// write metadata of the class to file
			file << "----------------Saved Model----------------\n";
			file << "Type: " << ((matrix_flag_) ? "2D" : (tensor_3d_flag_) ? "3D" : "4D") << "\n";
			file << "Size: (" << size_ << ")\n";
			file << "Cost_function: " << cost_function_ << "\n";

			// loop through network and call save method
			std::for_each(network_.begin(), network_.end(), [&file](auto &layer) { layer->save(file); });
			file.close();
		}

		/* 
		 * Load Model Method 
		*/

		void load_model(const std::string file_path, const int epochs = 1, const int batch_size = 10) {
			assert(!compile_flag_ && std::filesystem::exists(file_path));

			std::string line;						// empty string for each line in the file
			std::ifstream file;						// file object
			std::istringstream phrase;				// for individual phrase or string separated by blank

			file.open(file_path);					// open path
			std::getline(file, line);				// skip first line
			LoadModelParam(file, line, phrase);		// load the parameters of the model
			
			int j = 0;								// counter for current layer
			epochs_ = epochs;						// save epochs
			batch_size_ = batch_size;				// save batch size

			while (std::getline(file, line)) {					// begin parsing for each layer in the model
				if (!line.empty()) {

					if (line == "Dense") {
						LoadDense(file, line, phrase, j);
					
					} else if (line == "Embedding") {
						LoadEmbedding(file, line, phrase, j);
					
					} else if (line == "Distributed Dense") {
						LoadDistDense(file, line, phrase, j);
					
					} else if (line == "LSTM") {
						LoadLSTM(file, line, phrase, j);
					
					} else if (line == "RNN") {
						LoadRNN(file, line, phrase, j);
					
					} else if (line == "Conv2D") {
						LoadConv2D(file, line, phrase, j);
					
					} else if (line == "Pool2D") {
						LoadPool2D(file, line, phrase, j);
					
					} else if (line == "Flat2D") {
						LoadFlat2D(file, line, phrase, j);
					
					} else if (line == "Dense2D") {
						LoadDense2D(file, line, phrase, j);
					}

				}
			}
			
			reload_flag_ = true;
			file.close();
		}

	private:
		std::string cost_function_;													// cost function type
		
		float accuracy_ = 0.0, loss_ = 0.0;					    				    // performance metadata of the model
		
		int size_ = 0, batch_size_, epochs_;										// num. of layers, batch size, training pts.
		
		std::vector<int> reset_indexes_;											// vector of sequence layers' index for reseting their previous parameters

		std::vector<std::unique_ptr<Layer>> network_;							    // neural network

		bool compile_flag_ = false, matrix_flag_ = false, reload_flag_ = false, 
			 tensor_3d_flag_ = false, tensor_4d_flag_ = false;						// flags for appropriate checkings
		
		/*
		 * Initalize the input shape of layers in the network
		*/

		void SetLayersInput(const Eigen::DSizes<ptrdiff_t, 2> input_shape) {
			network_[0]->init(input_shape);
			for (int j = 1; j < size_; j ++) {
				network_[j]->init(network_[j - 1]->Get2dOutputShape());
			}
		}

		void SetLayersInput(const Eigen::DSizes<ptrdiff_t, 3> input_shape) {
			network_[0]->init(input_shape);
			for (int j = 1; j < size_; j ++) {
				network_[j]->init(network_[j - 1]->Get3dOutputShape());
			}
		}

		void SetLayersInput(const Eigen::DSizes<ptrdiff_t, 4> input_shape) {
			network_[0]->init(input_shape);
			for (int j = 1; j < size_; j ++) {
				network_[j]->init(network_[j - 1]->Get4dOutputShape());
			}
		}

		void SetResetVector() {
			for (int j = 0; j < size_; j ++) {
				if (network_[j]->name() == "LSTM" || network_[j]->name() == "RNN" || network_[j]->name() == "GRU" ||
					network_[j]->name() == "BiLSTM" || network_[j]->name() == "BiRNN") {
					
					reset_indexes_.push_back(j);
				}
			}
		}

		const float rand() {
			std::random_device rd;
			std::mt19937 gen(rd());
			std::uniform_real_distribution <float> dist (0.0, 1.0);
			return dist(gen);
		}

		/*
		 * Forward Propagation
		*/

		void Forward(const Eigen::MatrixXf &X, const Eigen::MatrixXf &Y) {
			network_[0]->forward(X);
			for (int j = 1; j < size_; j ++) {
				network_[j]->forward(network_[j - 1]->Get2dLayerH());
			}

			loss_ += cost_function_ == "mean_squared_error" ? network_.back()->MeanSquaredError(Y, accuracy_) : network_.back()->CrossEntropyError(Y, accuracy_);
		}

		void Forward(const Eigen::Tensor<float, 3> &X, const Eigen::Tensor<float, 3> &Y) {
            network_[0]->forward(X);
            for (int j = 1; j < size_; j ++) {
                network_[j]->forward(network_[j - 1]->Get3dLayerH());
            }

			loss_ += cost_function_ == "mean_squared_error" ? network_.back()->MeanSquaredError(Y, accuracy_) : network_.back()->CrossEntropyError(Y, accuracy_);
        }

		void Forward(const Eigen::Tensor<float, 4> &X, const Eigen::Tensor<float, 4> &Y) {
            network_[0]->forward(X);
            for (int j = 1; j < size_; j ++) {
				network_[j]->forward(network_[j - 1]->Get4dLayerH());
            }

			loss_ += cost_function_ == "mean_squared_error" ? network_.back()->MeanSquaredError(Y, accuracy_) : network_.back()->CrossEntropyError(Y, accuracy_);
        }

		/*
		 * Backward Propagation
		 * Update last layer
		 * Loop from L - 2 to first layer
		 * Retrieve gradient from previous layer and update 
		*/

		void Backward2D() {
			network_.back()->update();
			for (int j = size_ - 2; j >= 0; j --) {
				network_[j]->set_delta(network_[j + 1]->Get2dLayerDelta());
				network_[j]->update();
			}
		}

		void Backward3D() {
			network_.back()->update();
			for (int j = size_ - 2; j >= 0; j --) {
				network_[j]->set_delta(network_[j + 1]->Get3dLayerDelta());
				network_[j]->update();
			}
		}

		void Backward4D() {
			network_.back()->update();
			for (int j = size_ - 2; j >= 0; j --) {
				network_[j]->set_delta(network_[j + 1]->Get4dLayerDelta());
				network_[j]->update();
			}
		}


		/*
		 *
		 * Parse Method for a variety of array structures
		 * Load parsed number into the designated parameter
		 *
		 */

		void ParseArray(Eigen::DSizes<ptrdiff_t, 2> &target, const std::string &str_array) {
			int index = 0;
			std::string str_num; 
			
			for (int j = 1; j < str_array.size(); j ++) {
				// if character is not a comma or closed bracket, add it to the string
				str_num = str_array[j] != ',' && str_array[j] != ']' ? str_num + str_array[j] : str_num;
				
				if (str_array[j] == ',' || str_array[j] == ']') {
					target[index] = stoi(str_num);
					str_num = "";
					index ++;
				}
			}
		}

		void ParseArray(Eigen::DSizes<ptrdiff_t, 3> &target, const std::string &str_array) {
			int index = 0;
			std::string str_num; 
			for (int j = 1; j < str_array.size(); j ++) {
				// if character is not a comma or closed bracket, add it to the string
				str_num = str_array[j] != ',' && str_array[j] != ']' ? str_num + str_array[j] : str_num;

				if (str_array[j] == ',' || str_array[j] == ']') {
					target[index] = stoi(str_num);
					str_num = "";
					index ++;
				}
			}
		}
		
		void ParseArray(Eigen::DSizes<ptrdiff_t, 4> &target, const std::string &str_array) {
			int index = 0;
			std::string str_num; 
			
			for (int j = 1; j < str_array.size(); j ++) {
				// if character is not a comma or closed bracket, add it to the string
				str_num = str_array[j] != ',' && str_array[j] != ']' ? str_num + str_array[j] : str_num;
				if (str_array[j] == ',' || str_array[j] == ']') {
					target[index] = stoi(str_num);
					str_num = "";
					index ++;
				}
			}
		}

		void ParseArray(std::vector<float> &target, const std::string &str_array) {
			int index = 0;
			std::string str_num;

			for (int j = 1; j < str_array.size(); j ++) {
				// if character is not a comma or closed bracket, add it to the string
				str_num = str_array[j] != ',' && str_array[j] != ']' ? str_num + str_array[j] : str_num;
				if (str_array[j] == ',' || str_array[j] == ']') {
					target[index] = stof(str_num);
					str_num = "";
					index ++;
				}
			}
		}	

		void ParseArray(std::vector<std::vector<float>> &target, const std::string &str_array) {
			int row = 0, col = 0;
			std::string str_num;
			
			for (int j = 2; j < str_array.size() - 1; j ++) {
				// if character is not a comma or open and closed brackets, add it to the string
				str_num = str_array[j] != ',' && str_array[j] != '[' && str_array[j] != ']' ? str_num + str_array[j] : str_num;
				
				if (str_array[j] == '[') {
					row ++;

				} else if (str_array[j] == ',' && str_num != "") {
					target[row][col] = stof(str_num);
					str_num = "";
					col ++;

				} else if (str_array[j] == ']') {
					target[row][col] = stof(str_num);
					str_num = "";
					col = 0; 
				}
			}
		}
		

		/*
		 *
		 * Next Parameter Method
		 * Retrieve the next parameter in the file
		 *
		 */

		void next_param(std::ifstream &file, std::string &line, std::string &current, std::istringstream &phrase) {
			std::getline(file, line); phrase = std::istringstream(line);
			phrase >> current; current = ""; phrase >> current;
		}

		void next_param(int &value, std::ifstream &file, std::string &line, std::string &current, std::istringstream &phrase) {
			std::getline(file, line); phrase = std::istringstream(line);
			phrase >> current; current = ""; phrase >> current;
			value = stoi(current.substr(1, current.size() - 2));
		}

		void next_param(float &value, std::ifstream &file, std::string &line, std::string &current, std::istringstream &phrase) {
			std::getline(file, line); phrase = std::istringstream(line);
			phrase >> current; current = ""; phrase >> current;
			value = stof(current.substr(1, current.size() - 2));
		}
		
		void next_param(std::string &value, std::ifstream &file, std::string &line, std::string &current, std::istringstream &phrase) {
			std::getline(file, line); phrase = std::istringstream(line);
			phrase >> current; current = ""; phrase >> current;
			value = current;
		}

		void next_param(std::vector<float> &value, std::ifstream &file, std::string &line, std::string &current, std::istringstream &phrase) {
			std::getline(file, line); phrase = std::istringstream(line);
			phrase >> current; current = ""; phrase >> current;
			ParseArray(value, current);
		}
		
		void next_param(std::vector<std::vector<float>> &value, std::ifstream &file, std::string &line, std::string &current, std::istringstream &phrase) {
			std::getline(file, line); phrase = std::istringstream(line);
			phrase >> current; current = ""; phrase >> current;
			ParseArray(value, current);
		}
		
		void next_param(Eigen::DSizes<ptrdiff_t, 2> &value, std::ifstream &file, std::string &line, std::string &current, std::istringstream &phrase) {
			std::getline(file, line); phrase = std::istringstream(line);
			phrase >> current; current = ""; phrase >> current;
			ParseArray(value, current);
		}
		
		void next_param(Eigen::DSizes<ptrdiff_t, 3> &value, std::ifstream &file, std::string &line, std::string &current, std::istringstream &phrase) {
			std::getline(file, line); phrase = std::istringstream(line);
			phrase >> current; current = ""; phrase >> current;
			ParseArray(value, current);
		}
		
		void next_param(Eigen::DSizes<ptrdiff_t, 4> &value, std::ifstream &file, std::string &line, std::string &current, std::istringstream &phrase) {
			std::getline(file, line); phrase = std::istringstream(line);
			phrase >> current; current = ""; phrase >> current;
			ParseArray(value, current);
		}


		/* 
		 *
		 * Parameter Retrieval Methods Section
		 * Retrieve parameters of the designated layer (indicated by the method name)
		 * Create the unique pointer to the layer and set its parameters
		 * Call the layer's load method 
		 * Increment the Layer counter (j)
		 *
		 */

		void LoadModelParam(std::ifstream &file, std::string &line, std::istringstream &phrase) {
			std::string current;
			next_param(file, line, current, phrase);
			matrix_flag_ = current == "2D" ? true : matrix_flag_;
		    tensor_3d_flag_ = current == "3D" ? true : tensor_3d_flag_;
			tensor_4d_flag_ = current == "4D" ? true : tensor_4d_flag_;
			
			next_param(size_, file, line, current, phrase);
			next_param(cost_function_, file, line, current, phrase);
			network_.resize(size_);
		}

		void LoadDense(std::ifstream &file, std::string &line, std::istringstream &phrase, int &j) {
			assert(matrix_flag_ && !tensor_3d_flag_ && !tensor_4d_flag_ && j < size_);
			
			float lrate, erate;
			std::string current, activation;
			Eigen::DSizes<ptrdiff_t, 2> input_shape, output_shape;

			next_param(activation, file, line, current, phrase);
			next_param(lrate, file, line, current, phrase);
			next_param(erate, file, line, current, phrase);
			next_param(input_shape, file, line, current, phrase);
			next_param(output_shape, file, line, current, phrase);

			std::vector<float> bias (output_shape[0] * output_shape[1], 0.0), weight (input_shape[1] * output_shape[1], 0.0);
			next_param(bias, file, line, current, phrase);
			next_param(weight, file, line, current, phrase);

			network_[j] = std::move(std::unique_ptr<Layer> {new Dense (output_shape[1], lrate, activation, erate)});
			network_[j]->load(input_shape, output_shape, bias, weight);	
			j ++;
		}
		
		void LoadEmbedding(std::ifstream &file, std::string &line, std::istringstream &phrase, int &j) {
			assert(matrix_flag_ && !tensor_3d_flag_ && !tensor_4d_flag_ && j < size_);
			
			int lexicon_space;
			float lrate, erate;
			std::string current; 
			Eigen::DSizes<ptrdiff_t, 2> input_shape, output_shape;
		
			next_param(lexicon_space, file, line, current, phrase);
			next_param(lrate, file, line, current, phrase);
			next_param(erate, file, line, current, phrase);
			next_param(input_shape, file, line, current, phrase);
			next_param(output_shape, file, line, current, phrase);

			std::vector<std::vector<float>> weight (lexicon_space, std::vector<float> (input_shape[1] * output_shape[1], 0.0)); 
			next_param(weight, file, line, current, phrase);
			
			network_[j] = std::move(std::unique_ptr<Layer> {new Embedding (lexicon_space, output_shape[1], lrate, erate)});	
			network_[j]->load(input_shape, output_shape, weight);
			j ++;
		}

		void LoadDistDense(std::ifstream &file, std::string &line, std::istringstream &phrase, int &j) {
			assert(matrix_flag_ && !tensor_3d_flag_ && !tensor_4d_flag_ && j < size_);
			
			float lrate, erate;
			std::string current, activation;
			Eigen::DSizes<ptrdiff_t, 2> input_shape, output_shape;
			
			next_param(activation, file, line, current, phrase);
			next_param(lrate, file, line, current, phrase);
			next_param(erate, file, line, current, phrase);
			next_param(input_shape, file, line, current, phrase);
			next_param(output_shape, file, line, current, phrase);

			std::vector<float> bias (output_shape[0] * output_shape[1], 0.0), weight (input_shape[1] * output_shape[1], 0.0);
			next_param(bias, file, line, current, phrase);
			next_param(weight, file, line, current, phrase);

			network_[j] = std::move(std::unique_ptr<Layer> {new DistributedDense (output_shape[1], lrate, activation, erate)});
			network_[j]->load(input_shape, output_shape, bias, weight);	
			j ++;
		}

		void LoadLSTM(std::ifstream &file, std::string &line, std::istringstream &phrase, int &j) {
			assert(matrix_flag_ && !tensor_3d_flag_ && !tensor_4d_flag_ && j < size_);
			
			float lrate, erate;
			std::string current, activation, recurrent_activation;
			Eigen::DSizes<ptrdiff_t, 2> input_shape, output_shape;

			next_param(activation, file, line, current, phrase);
			next_param(recurrent_activation, file, line, current, phrase);
			next_param(lrate, file, line, current, phrase);
			next_param(erate, file, line, current, phrase);
			next_param(input_shape, file, line, current, phrase);
			next_param(output_shape, file, line, current, phrase);

			std::vector<float> weight_fgate(input_shape[1] * output_shape[1], 0.0),
							   weight_igate(input_shape[1] * output_shape[1], 0.0),
							   weight_ogate(input_shape[1] * output_shape[1], 0.0),
							   weight_cgate(input_shape[1] * output_shape[1], 0.0),
							   u_weight_fgate(output_shape[1] * output_shape[1], 0.0),
							   u_weight_igate(output_shape[1] * output_shape[1], 0.0),
							   u_weight_ogate(output_shape[1] * output_shape[1], 0.0),
							   u_weight_cgate(output_shape[1] * output_shape[1], 0.0),
							   bias_fgate (output_shape[1], 0.0), bias_igate (output_shape[1], 0.0),
							   bias_ogate (output_shape[1], 0.0), bias_cgate (output_shape[1], 0.0);

			next_param(bias_fgate, file, line, current, phrase);
			next_param(weight_fgate, file, line, current, phrase);
			next_param(u_weight_fgate, file, line, current, phrase);

			next_param(bias_igate, file, line, current, phrase);
			next_param(weight_igate, file, line, current, phrase);
			next_param(u_weight_igate, file, line, current, phrase);
			
			next_param(bias_ogate, file, line, current, phrase);
			next_param(weight_ogate, file, line, current, phrase);
			next_param(u_weight_ogate, file, line, current, phrase);

			next_param(bias_cgate, file, line, current, phrase);
			next_param(weight_cgate, file, line, current, phrase);
			next_param(u_weight_cgate, file, line, current, phrase);
			
			network_[j] = std::unique_ptr<Layer> {new LSTM(output_shape[1], lrate, activation, recurrent_activation, erate)};
			network_[j]->load(input_shape, output_shape, 
							  bias_fgate, weight_fgate, u_weight_fgate,
							  bias_igate, weight_igate, u_weight_igate,
							  bias_ogate, weight_ogate, u_weight_ogate,
							  bias_cgate, weight_cgate, u_weight_cgate);
			j ++;
		}

		void LoadRNN(std::ifstream &file, std::string &line, std::istringstream &phrase, int &j) {
			assert(matrix_flag_ && !tensor_3d_flag_ && !tensor_4d_flag_ && j < size_);
			
			float lrate, erate; 
			std::string current, activation, recurrent_activation;
            Eigen::DSizes<ptrdiff_t, 2> input_shape, output_shape;

			next_param(activation, file, line, current, phrase);
			next_param(recurrent_activation, file, line, current, phrase);
			next_param(lrate, file, line, current, phrase);
			next_param(erate, file, line, current, phrase);
			next_param(input_shape, file, line, current, phrase);
			next_param(output_shape, file, line, current, phrase);
			
			std::vector<float> bias_igate (output_shape[1], 0.0), 
							   weight_igate (input_shape[1] * output_shape[1], 0.0),
							   u_weight_igate (output_shape[1] * output_shape[1], 0.0),
							   bias_h (output_shape[1], 0.0), 
							   weight_h (output_shape[1] * output_shape[1], 0.0);

			next_param(bias_igate, file, line, current, phrase);
			next_param(weight_igate, file, line, current, phrase);
			next_param(u_weight_igate, file, line, current, phrase);
			next_param(bias_h, file, line, current, phrase);
			next_param(weight_h, file, line, current, phrase);

			network_[j] = std::unique_ptr<Layer> {new RNN (output_shape[1], lrate, activation, recurrent_activation, erate) };
			network_[j]->load(input_shape, output_shape, bias_igate, weight_igate, u_weight_igate, bias_h, weight_h);
			j ++;
		}

		void LoadConv2D(std::ifstream &file, std::string &line, std::istringstream &phrase, int &j) {
			assert(!matrix_flag_ && tensor_3d_flag_ && !tensor_4d_flag_ && j < size_);

			float lrate, erate;
			std::string current, activation;
			int width_stride, height_stride;
			Eigen::DSizes<ptrdiff_t, 4> kernel_shape;
			Eigen::DSizes<ptrdiff_t, 3> input_shape, output_shape;
			
			next_param(activation, file, line, current, phrase);
			next_param(lrate, file, line, current, phrase);
			next_param(erate, file, line, current, phrase);
			next_param(height_stride, file, line, current, phrase);
			next_param(width_stride, file, line, current, phrase);
			next_param(input_shape, file, line, current, phrase);
			next_param(output_shape, file, line, current, phrase);
			next_param(kernel_shape, file, line, current, phrase);
			
			std::vector <float> bias (kernel_shape[0], 0.0), weight (kernel_shape[0] * kernel_shape[1] * kernel_shape[2] * kernel_shape[3], 0.0);
			next_param(bias, file, line, current, phrase);
			next_param(weight, file, line, current, phrase);

			network_[j] = std::unique_ptr<Layer> {new Conv2D (kernel_shape[0], kernel_shape[2], kernel_shape[3], height_stride, width_stride, lrate, activation, erate)}; 
			network_[j]->load(input_shape, output_shape, bias, weight);
			j ++;
		}

		void LoadPool2D(std::ifstream &file, std::string &line, std::istringstream &phrase, int &j) {
			assert(!matrix_flag_ && tensor_3d_flag_ && !tensor_4d_flag_ && j < size_);

			std::string current;
			int width_stride, height_stride;
			Eigen::DSizes<ptrdiff_t, 3> input_shape, output_shape, kernel_shape;

			next_param(height_stride, file, line, current, phrase);
			next_param(width_stride, file, line, current, phrase);
			next_param(input_shape, file, line, current, phrase);
            next_param(output_shape, file, line, current, phrase);
            next_param(kernel_shape, file, line, current, phrase);

			network_[j] = std::unique_ptr<Layer> {new Pool2D (kernel_shape[1], kernel_shape[2], height_stride, width_stride)};
			network_[j]->load(input_shape, output_shape);
			j ++; 
		}

		void LoadDense2D(std::ifstream &file, std::string &line, std::istringstream &phrase, int &j) {
			assert(!matrix_flag_ && tensor_3d_flag_ && !tensor_4d_flag_ && j < size_);

			float lrate, erate;
            std::string current, activation;
			Eigen::DSizes<ptrdiff_t, 3> input_shape, output_shape;
  
            next_param(activation, file, line, current, phrase);
            next_param(lrate, file, line, current, phrase);
            next_param(erate, file, line, current, phrase);
			next_param(input_shape, file, line, current, phrase);
            next_param(output_shape, file, line, current, phrase);

			std::vector <float> bias (output_shape[1] * output_shape[2], 0.0), weight (input_shape[2] * output_shape[2], 0.0);
			next_param(bias, file, line, current, phrase);
			next_param(weight, file, line, current, phrase); 

			network_[j] = std::unique_ptr<Layer> {new Dense2D (output_shape[2], lrate, activation, erate)};
			network_[j]->load(input_shape, output_shape, bias, weight);
			j ++;
		}
		
		void LoadFlat2D(std::ifstream &file, std::string &line, std::istringstream &phrase, int &j) {
			assert(!matrix_flag_ && tensor_3d_flag_ && !tensor_4d_flag_ && j < size_);

			std::string current;
			Eigen::DSizes<ptrdiff_t, 3> input_shape, output_shape;
			
			next_param(input_shape, file, line, current, phrase);
            next_param(output_shape, file, line, current, phrase);

			network_[j] = std::unique_ptr<Layer> {new Flat2D ()};
			network_[j]->load(input_shape, output_shape);
			j ++;
		}

};

}
