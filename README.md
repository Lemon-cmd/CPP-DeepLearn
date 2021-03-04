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

__Classes__ | __Methods__
------------ | -------------
__Sequential__ | __Sequential__() <br /> __summary__() <br /> __add__(std::unique_ptr<Layer> &&) <br /> __add__(std::unique_ptr<Layer> &) <br /> __compile__(const Eigen::DSizes<ptrdiff_t, 2>, const int, const int, const std::string) <br /> __compile__(const Eigen::DSizes<ptrdiff_t, 3>, const int, const int, const std::string) <br /> __compile__(const Eigen::DSizes<ptrdiff_t, 4>, const int, const int, const std::string) <br /> __fit__(const std::vector<Eigen::MatrixXf> &, const std::vector<Eigen::MatrixXf> &) <br /> __fit__(const std::vector<Eigen::Tensor<float, 3>> &, const std::vector<Eigen::Tensor<float, 3>> &) <br /> __fit__(const std::vector<Eigen::Tensor<float, 4>> &, const std::vector<Eigen::Tensor<float, 4>> &) 
