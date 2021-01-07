# CPP-DeepLearning
Template for C++ neural network variants built via Eigen Library


**Usage**
In order to create a network, one must first construct the network object.

```
   std::unique_ptr<network> net (new network()); 
```
or 
```
   network net ();
```
or 
```
   network* net = new network();
```

The parameter of the ***network*** object is left blank as the optional cost function parameter (string) is set as mean-square-error.\
The user can set the parameter as "mean-square-error" or "cross-entropy-error". (See template **network.hpp**)   

e.g.,
```
   std::unique_ptr<network> net (new network("cross-entropy-error")); 
```

To create layers of the network, one must create their pointer to the abstract ***Layer*** class while setting it to the desired object.
    
e.g.,
```
  Layer* l0 = new Embedding<521, 1, 20>();
  Layer* l1 = new LSTM<150, 100, 20>();
  Layer* l2 = new Dense<521, 150, 20>("softmax");
  net->add(l0); net->add(l1); net->add(l2);
```

For further details, take a look at the ***main.cpp*** file

To build the code, two files are included:     
***precomp.sh*** for precompiling the header files     
***build.sh*** to build the main.cpp file     
    
**Tips**
Two template for data processing are included:     ***parser.hpp***    ***parse_text.hpp***
The first only parses datasets with many features. (X0, X1, ..., Xn). The template provides two parameters <train_percent, num_Ys> for parsing.

```
  parse<75, 1> data ("./data/Tweets_txt.csv", "./data/Tweets_sentiment.csv");
```

The later focuses on corpora, where it contains two parameters <train_percent, sequence_length> for parsing the dataset.
Additionally, the constructor has two parameters (path, split=false).

```
  parse_text<25, 20> data ("./data/sly_fox.txt");
```
```
  parse_text<25, 20> data ("./data/sly_fox.txt", false);
```

For further details, take a look at these files to make appropriate changes. Data processing is a tiring, inconsistent process.

**Common Pitfalls**
The provided classes make great use of templates, which required knowing their template values ahead of compile time.      

***parse*** class does provide a method to display the number of features. 
```
  data.feature_size();
``` 

Meanwhile, ***parse_text*** provides the ***vsize()*** method to get the number of features or vocabulary size for text analysis.   

If the user wants to add custom activation functions, please do so in the template **layer.hpp** and in the Layer class for the function to be shared with other layer variants.



