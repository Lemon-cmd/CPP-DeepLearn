# CPP-DeepLearning
Template for C++ neural network variants built using the Eigen Library


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
The user can set the parameter as "mean_square_error" or "categorical_cross_entropy". (See template **network.hpp**)   

e.g.,
```
   std::unique_ptr<network> net (new network("categorical_cross_entropy")); 
```

To create layers of the network, one must create a unique pointer to the abstract ***Layer*** class while setting it to the desired object.
    
e.g.,
```
   net.add(std::unique_ptr<Layer> (new embedding <560, 100, 100>()));
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



