# CPP-DeepLearn

__DPP__ is a C++ deep learning library built using Eigen and thread library. It provides an easy way to create neural networks similar to that of keras. 

## Usage    
To create a model, one must first construct the __sequential__ class.

```c++
   using namespace dpp;
   
   std::unique_ptr<Sequential> model {new Sequential()}; 
   
   // or
   Sequential model;
```

To add a layer to the model, one must create a unique pointer to the abstract __Layer__ class while setting it to the desired object.
    
```c++
   #define L std::unique_ptr<Layer>  
   
   model->add(L {new Embedding (250, 100, 0.001, 1e-8)});
   
   model->add(L {new LSTM (2500)});
```

Once all layers are __defined__ and the model is wished to be __trained__, it is __required__ to first be __compiled__.   

```c++
   #define dim2d Eigen::DSizes<ptrdiff_t, 2> 
   
   // input shape, epochs = optional (default to 1), batch size = optional, cost function = optional (default to mean_squared_error)
   model->compile(dim2d {1, 100}, 100, 100, "cross_entropy_error");        
   
   model->fit(X_train, Y_train);                                           // train
   
   model->evaluate(X_test, Y_test)                                         // test
```

Layers can be created separately as well and are required to be __initialized__ first before trainning.

```c++
   // Dense (neurons, learning rate, activation, optimization rate) 
   L l0 { new Dense (100, 0.001, "relu", 1e-8) };             
   
   // LSTM (neurons, learning rate, activation, recurrent activation, optimization rate)
   L l1 { new LSTM (100) };                                 
   
   l0->init(dim2d {1, 100});           // init requires input shape 
   
   l1->init(l0->Get2dOutputShape());
```
__Training__ can be done in two ways: __sequence__ or __non-sequence__

```c++
   model->fit(X_train, Y_train);                // non-sequence training
   
   model->fit_sequence(X_train, Y_train);       // sequence training
```

Once training is done, it is possible to __save__ the model. However, it must be __compiled__.

```c++
   model->compile(....);
   
   model->save_model("saved_model.txt");                 // good
   
   Sequential new_model;
   
   new_model.save_model("new_saved_model.txt");          // error
 ```

It is also possible to load a pre-trained model as well but requiring that the object is not __compiled__.

```c++
   // file path,  epochs = optional, batch size = optional
   model->load_model("saved_model.txt", 1000, 1000);        
 ```

A predict method is included as well.

```c++
   model->predict(sentence, expect_result);
```


## Available Classes

|__Classes__ | __Methods__ | 
|:----------- |-------------|
|__Sequential__ | __Sequential__() <br /> __void summary__() <br /> __void add__(std::unique_ptr<Layer> &&) <br /> __void add__(std::unique_ptr<Layer> &) <br /> __void compile__(const Eigen::DSizes<ptrdiff_t, 2>, const int, const int, const std::string) <br /> __void compile__(const Eigen::DSizes<ptrdiff_t, 3>, const int, const int, const std::string) <br /> __void compile__(const Eigen::DSizes<ptrdiff_t, 4>, const int, const int, const std::string) <br /> __void fit__(const std::vector<Eigen::MatrixXf> &, const std::vector<Eigen::MatrixXf> &) <br /> __void fit__(const std::vector<Eigen::Tensor<float, 3>> &, const std::vector<Eigen::Tensor<float, 3>> &) <br /> __void fit__(const std::vector<Eigen::Tensor<float, 4>> &, const std::vector<Eigen::Tensor<float, 4>> &) <br /> __void fit_sequence__(const std::vector<Eigen::MatrixXf> &, const std::vector<Eigen::MatrixXf> &) <br /> __void evaluate__(const std::vector<Eigen::MatrixXf> &, const std::vector<Eigen::MatrixXf> &) <br /> __void evaluate__(const std::vector<Eigen::Tensor<float, 3>> &, const std::vector<Eigen::Tensor<float, 3>> &) <br /> __void evaluate__(const std::vector<Eigen::Tensor<float, 4>> &, const std::vector<Eigen::Tensor<float, 4>> &) <br /> __void predict__(const Eigen::MatrixXf &, const Eigen::MatrixXf &) <br /> __void predict__(const Eigen::Tensor<float, 3> &, const Eigen::Tensor<float, 3> &) <br /> __void predict__(const Eigen::Tensor<float, 4> &, const Eigen::Tensor<float, 4> &) <br /> __void save_model__(const std::string) <br /> __void load_model__(const std::string, const int, const int)|
|__Dense__ | __Dense__(const int, const float, const std::string, const float) <br /> __void init__(const Eigen::DSizes<ptrdiff_t, 2>) <br /> __void topology()__ <br />  __void forward__(const Eigen::MatrixXf &) <br /> __void update__() <br /> __void set_delta__(const Eigen::MatrixXf &) <br /> __const std::string name__() <br /> __const std::string type__() <br /> __const float MeanSquaredError__(const Eigen::MatrixXf &, float &) <br /> __const float CrossEntropyError__(const Eigen::MatrixXf &, float &) <br /> __void save__(std::ofstream &) <br /> __void load__(const Eigen::DSizes<ptrdiff_t, 2> &, const Eigen::DSizes<ptrdiff_t, 2> &, const std::vector \<float> &, const std::vector \<float> &)|
|__Embedding__ | __Embedding__(const int, const int, const float, const float) <br /> __void init__(const Eigen::DSizes<ptrdiff_t, 2>) <br /> __void topology()__ <br /> __void forward__(const Eigen::MatrixXf &) <br /> __void update__() <br /> __void set_delta__(const Eigen::MatrixXf &) <br /> __const std::string name__() <br /> __const std::string type__() <br /> __const float MeanSquaredError__(const Eigen::MatrixXf &, float &) <br /> __void save__(std::ofstream &) <br /> __void load__(const Eigen::DSizes<ptrdiff_t, 2> &, const Eigen::DSizes<ptrdiff_t, 2> &, const std::vector \<float> &, const std::vector \<float> &)|
| __LSTM__ | __LSTM__(const int, const float, const std::string, const float) <br /> __void init__(const Eigen::DSizes<ptrdiff_t, 2>) <br /> __void topology()__ <br /> __void forward__(const Eigen::MatrixXf &) <br /> __void update__() <br /> __void set_delta__(const Eigen::MatrixXf &) <br /> __const std::string name__() <br /> __const std::string type__() <br /> __const float MeanSquaredError__(const Eigen::MatrixXf &, float &) <br /> __void save__(std::ofstream &) <br /> __void load__(const Eigen::DSizes<ptrdiff_t, 2> &, const Eigen::DSizes<ptrdiff_t, 2> &, const std::vector \<float> &, const std::vector \<float> &)|
| __RNN__ | __RNN__(const int, const float, const std::string, const float) <br /> __void init__(const Eigen::DSizes<ptrdiff_t, 2>) <br /> __void topology()__ <br /> __void forward__(const Eigen::MatrixXf &) <br /> __void update__() <br /> __void set_delta__(const Eigen::MatrixXf &) <br /> __const std::string name__() <br /> __const std::string type__() <br /> __const float MeanSquaredError__(const Eigen::MatrixXf &, float &) <br /> __void save__(std::ofstream &) <br /> __void load__(const Eigen::DSizes<ptrdiff_t, 2> &, const Eigen::DSizes<ptrdiff_t, 2> &, const std::vector \<float> &, const std::vector \<float> &)|
| __DistributedDense__ | __DistributedDense__(const int, const float, const std::string, const float) <br /> __void init__(const Eigen::DSizes<ptrdiff_t, 2>) <br /> __void topology()__ <br />  __void forward__(const Eigen::MatrixXf &) <br /> __void update__() <br /> __void set_delta__(const Eigen::MatrixXf &) <br /> __const std::string name__() <br /> __const std::string type__() <br /> __const float MeanSquaredError__(const Eigen::MatrixXf &, float &) <br /> __const float CrossEntropyError__(const Eigen::MatrixXf &, float &) <br /> __void save__(std::ofstream &) <br /> __void load__(const Eigen::DSizes<ptrdiff_t, 2> &, const Eigen::DSizes<ptrdiff_t, 2> &, const std::vector \<float> &, const std::vector \<float> &)|
| __Conv2D__ | __Conv2D__(const int = 1, const int = 1, const int = 1, const int = 1, const int = 1, const float = 0.001, const std::string = "normal", const float = 1e-8) <br /> __void init__(const Eigen::DSizes<ptrdiff_t, 3>) <br /> __void forward__(const Eigen::Tensor<float, 3> &) <br /> __void update__() <br /> __void set_delta__(const Eigen::Tensor<float, 3> &) <br /> __const std::string name()__ <br /> __const std::string type()__ <br /> __const float MeanSquaredError__(const Eigen::Tensor<float, 3> &, float &) <br /> __void save__(std::ofstream &) <br /> __void load__(const Eigen::DSizes<ptrdiff_t, 3> &, const Eigen::DSizes<ptrdiff_t, 3> &, std::vector<float> &, std::vector<float> &)   
| __Pool2D__ | __Pool2D__()
 
