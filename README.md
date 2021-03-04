# CPP-DeepLearn

__DPP__ is a C++ deep learning library built using Eigen and thread library. It provides an easy way to create neural networks similar to that of keras. 

## Usage    
To create a model, one must first construct the __sequential__ class.

```c++
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

Once all layers are defined and the model is wished to train, it is required to first be compiled.   

```c++
   #define dim2d Eigen::DSizes<ptrdiff_t, 2> 
   
   model->compile(dim2d {1, 100}, 100, 100, "cross_entropy_error");
   model->fit(X_train, Y_train);       // train
   model->evaluate(X_test, Y_test)     // test
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
 
