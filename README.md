# CPP-DeepLearn

DPP is a deep learning library built in c++ using Eigen and thread library. It provides an easy way to create neural networks similar to that of keras. 

**Usage**     
To create a model, one must first construct the sequential object.

```c++
   std::unique_ptr<Sequential> model {new Sequential()}; 
   // or
   Sequential model;
```

To create layers of the network, one must create a unique pointer to the abstract ***Layer*** class while setting it to the desired object.
    
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

_Classes_ | _Methods_
------------ | -------------
Sequential | Sequential() <br /> summary() <br /> add(std::unique_ptr<Layer> &&) <br /> add(std::unique_ptr<Layer> &)
